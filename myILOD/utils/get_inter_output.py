import enum
import os
import logging
from collections import OrderedDict
from random import randint
from urllib import response
from detectron2.utils.collect_env import detect_compute_compatibility
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.config import get_cfg

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, verify_results, PascalVOCDetectionEvaluator
from detectron2.modeling import GeneralizedRCNNWithTTA

from myILOD.utils.register import my_register
import detectron2.utils.comm as comm

from PIL import Image, ImageDraw
from detectron2.data.detection_utils import convert_PIL_to_numpy
import numpy as np
import torch, sys, random, json, logging, time, cv2
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils.logger import setup_logger

from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
    print_csv_format,
    verify_results,
)

class Trainer(DefaultTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)

        # Ein:
        self.memory = self.build_memory(cfg)
        self.old_model = self.build_model(cfg)
        self.old_model.load_state_dict(torch.load(cfg.MODEL.WEIGHTS)['model'])

    def run_step(self):

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        '''
        data: file_name, image, instances(num_instances, image_height, image_width, fields=[gt_boxes, gt_classes])
        old_response: n * instances when eval(), // (20*512 , 4*512) (loss_dict) when training
        new_response: 20*512 , 4*512
        proposals: Instances() * 2000
        '''
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        with torch.no_grad():
            old_rpn_logits, old_rpn_proposals,  _ = self.old_model(data)
        
        rpn_logits, rpn_proposals, loss_dict = self.model(data)

        distill_loss_dict = self.rpn_distill_losses(old_rpn_logits[0], old_rpn_proposals[0], rpn_logits[0], rpn_proposals[0])
        
        loss_dict.update(distill_loss_dict)
        print(loss_dict)

        # for each_old_prediction, each_img in zip(old_proposals, data):
        #     each_img['old_proposals'] = old_proposals
        #     each_img['loss_weight'] = torch.tensor([1] * len(each_img['instances']) + [s for s in each_old_prediction['instances'].get('scores')])
        #     each_img['instances'].set(
        #         'gt_boxes', 
        #         each_img['instances'].get('gt_boxes').cat(
        #                 [each_img['instances'].get('gt_boxes'), each_old_prediction['instances'].get('pred_boxes').to('cpu')]
        #             )
        #         )          
        #     each_img['instances'].set(
        #         'gt_classes', 
        #         torch.cat(
        #                 (each_img['instances'].get('gt_classes'), each_old_prediction['instances'].get('pred_classes').cpu())
        #             )
        #         )
        #     print(each_img)
        
        losses = sum(loss_dict.values())
        self.optimizer.zero_grad()
        losses.backward()
        
        # use a new stream so the ops don't wait for DDP
        with torch.cuda.stream(
            torch.cuda.Stream()
        ):
            metrics_dict = loss_dict
            metrics_dict["data_time"] = data_time
            self._write_metrics(metrics_dict)
            self._detect_anomaly(losses, loss_dict)

        self.optimizer.step()

    @classmethod
    def build_memory(cls, cfg):
        if cfg.DATASETS.MEMORY:
            with open(cfg.DATASETS.MEMORY) as f:
                memory_dict = dict(json.load(f))
            return memory_dict
        else:
            return None

    @classmethod
    def build_train_loader(cls, cfg):
        return super().build_train_loader(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return PascalVOCDetectionEvaluator(dataset_name) 

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
    
    @classmethod
    def rpn_distill_losses(cls, old_rpn_logits, old_rpn_proposals, rpn_logits, rpn_proposals):
        
        #logits loss
        filter_logits = torch.zeros(rpn_logits.shape).to('cuda')
        diff_logits = torch.max(old_rpn_logits - rpn_logits, filter_logits) 
        logits_loss = torch.mean(torch.mul(diff_logits, diff_logits))

        #box loss
        thresh = 0
        s = old_rpn_proposals.shape

        # a = torch.tensor([torch.mul(img, img) for b in (old_rpn_proposals - rpn_proposals) for img in b])
        mask_box = old_rpn_logits.clone()
        mask_box[old_rpn_logits > 0.7] = 1
        mask_box[old_rpn_logits <= 0.7] = 0
        diff_boxes = old_rpn_proposals - rpn_proposals

        diff_boxes = torch.tensor([
            [
                torch.tensor([ torch.sum(diff_box * diff_box)]) 
                for diff_box in each_img
            ] 
            for each_img in diff_boxes
        ]).to('cuda')

        box_loss = torch.mean(diff_boxes * mask_box)

        loss = {}
        loss['distill_rpn_box'] = box_loss
        loss['distill_rpn_logits'] = logits_loss
        return loss

    @classmethod
    def feature_distill_losses(cls, old_feature, feature):

        feature_distillation_loss = torch.abs(old_feature - feature)
        return torch.mean(feature_distillation_loss)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg() # 拷贝default config副本
    cfg.merge_from_file(args.config_file)   # 从config file 覆盖配置
    cfg.merge_from_list(args.opts)          # 从CLI参数 覆盖配置
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):

    # ZJW: Myregister
    my_register()

    cfg = setup(args)
    
    if args.eval_only:
        model = Trainer.build_model(cfg)

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    
    model = Trainer.build_model(cfg) 

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()



    data_loader = build_detection_train_loader(cfg)

    inter_output = {}
    for data in data_loader:
        rpn_logits, rpn_proposals, loss_dict = model(data)
        inter_output[data['file_name']] = {'rpn_logits': rpn_logits, 'rpn_proposals': rpn_proposals}
        torch.save(inter_output, 'inter_output')
        sys.exit(0)
    

if __name__ == "__main__":
    args = default_argument_parser().parse_known_args()[0]
    args.dist_url = 'tcp://127.0.0.1:{}'.format(randint(30000,50000))
    print("Command Line Args:", args)

    main(args)