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

        # self.old_model.eval()

        with torch.no_grad():
            old_rpn_logits, old_rpn_proposals, old_features,  _ = self.old_model(data)
        
        rpn_logits, rpn_proposals, features, loss_dict = self.model(data)

        distill_loss_dict = self.calc_proposal_distill_loss(old_rpn_logits[0], old_rpn_proposals[0], rpn_logits[0], rpn_proposals[0])

        print(distill_loss_dict)
        sys.exit(0)
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
    def calc_proposal_distill_loss(cls, old_rpn_logits, old_rpn_proposals, rpn_logits, rpn_proposals):
        
        # num_pos_obj = proposal1.objectness_logits.shape
        # filter = torch.zeros(num_pos_obj).to('cuda')
        # print(diff)
        # loss = torch.sum([i * i for d in diff for i in d if abs(i) > 1])
        # print(filter)
        # loss = torch.mean(torch.max(torch.tensor([torch.mean(abs(d)) for d in diff], requires_grad = True).to('cuda'), filter))

        filter = torch.zeros(rpn_logits.shape).to('cuda')

        #logits loss
        diff_logits = torch.max(old_rpn_logits - rpn_logits, filter) 
        logits_loss = torch.mean(torch.mul(diff_logits, diff_logits))

        #box loss
        diff_proposals = old_rpn_proposals - rpn_proposals
        filter_mask = old_rpn_logits.clone()
        filter_mask[old_rpn_logits > 0.2] = 1
        print(filter_mask)
        box_loss = 0

        loss = {}
        loss['distill_rpn_box'] = box_loss
        loss['distill_rpn_logits'] = logits_loss
        return loss

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

if __name__ == "__main__":
    args = default_argument_parser().parse_known_args()[0]
    args.dist_url = 'tcp://127.0.0.1:{}'.format(randint(30000,50000))
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )