import copy
import os
import logging
from collections import OrderedDict
from random import randint

from cv2 import randn
from detectron2.data.build import build_batch_data_loader, get_detection_dataset_dicts
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.samplers.distributed_sampler import TrainingSampler
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
import torch, sys, logging, time, random

import numpy as np

def filter_classes_instances(dataset_dicts, valid_classes):

    logger = logging.getLogger(__name__)
    logger.info("Valid classes: " + str(valid_classes))
    logger.info("Removing objects ...")
    
    for entry in copy.copy(dataset_dicts):
        annos = entry["annotations"]
        for annotation in copy.copy(annos):
            if annotation["category_id"] not in valid_classes:
                annos.remove(annotation)
        if len(annos) == 0:
            dataset_dicts.remove(entry)
    return dataset_dicts

def reweight(loss_dict, r):
    task_loss = sum(v for k, v in loss_dict.items() if 'dist' not in k)
    distill_loss = sum(v for k, v in loss_dict.items() if 'dist' in k)
    return (1 - r) * task_loss + r * distill_loss

class Trainer(DefaultTrainer):

    def __init__(self, cfg):

        super().__init__(cfg)
        
        md = self.model.module if comm.get_world_size() > 1 else self.model

        if cfg.IOD.DISTILL:
            sd = torch.load(cfg.MODEL.WEIGHTS, map_location='cpu')['model']  # why map_location leads to cruption
            md.load_state_dict(sd, strict = False)
            md.t_model.load_state_dict(sd)
        else:
            self.resume_or_load()

    def Mixup(self, data):

        # input data
        for each_img in data:
            if torch.rand(1) > self.cfg.IOD.MIXPRO:
                lambd = np.random.beta(5,5)
                img1 = each_img['image'].to('cuda')
                # memory data
                mm_data = random.choice(self.memory)
                img2= mm_data['image'].to('cuda')

                # operation
                height = max(img1.shape[1], img2.shape[1])
                width = max(img1.shape[2], img2.shape[2])

                mix_img = torch.zeros([3 ,height, width], device='cuda')
                mix_img[:, :img1.shape[1], :img1.shape[2]] = img1 * lambd
                mix_img[:, :img2.shape[1], :img2.shape[2]] += img2 * (1. - lambd)

                # fix
                each_img['image'] = mix_img
                each_img['instances']._fields['gt_boxes'].tensor = torch.cat((each_img['instances']._fields['gt_boxes'].tensor, mm_data['instances']._fields['gt_boxes'].tensor))
                each_img['instances']._fields['gt_classes'] = torch.cat((each_img['instances']._fields['gt_classes'], mm_data['instances']._fields['gt_classes']))
                each_img['loss_weight'] = lambd
            else:
                each_img['loss_weight'] = None

    def run_step(self):

        '''
        data: file_name, image, instances(num_instances, image_height, image_width, fields=[gt_boxes, gt_classes])
        old_response: n * instances when eval(), // (20*512 , 4*512) (loss_dict) when training
        new_response: 20*512 , 4*512
        proposals: Instances() * 2000
        '''
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._data_loader_iter)
        if self.cfg.IOD.MEMORY_AUG: self.Mixup(data)
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        losses = reweight(loss_dict, self.cfg.IOD.REWEIGHT)
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

    def build_train_loader(self, cfg):
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON
            else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )

        # ZJW
        print(len(dataset_dicts))
        if cfg.IOD.NEW_CLS:
            valid_classes = range(cfg.IOD.OLD_CLS, cfg.IOD.OLD_CLS + cfg.IOD.NEW_CLS)
            dataset_dicts = filter_classes_instances(dataset_dicts, valid_classes)        

        if cfg.IOD.MEMORY:
            memory_dicts = get_detection_dataset_dicts(
                    cfg.IOD.MEMORY,
                    filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                    min_keypoints=0,
                    proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None
                    )
            memory_dataset = DatasetFromList(memory_dicts)
            memory_dataset = MapDataset(memory_dataset, DatasetMapper(cfg, True))
            for m_d in memory_dicts: m_d['memory'] = True
            for d_d in dataset_dicts: d_d['memory'] = False
            dataset_dicts += memory_dicts
            self.memory = memory_dataset
        print(len(dataset_dicts))

        dataset = DatasetFromList(dataset_dicts, copy=False)
        mapper = DatasetMapper(cfg, True)
        dataset = MapDataset(dataset, mapper)

        sampler_name = cfg.DATALOADER.SAMPLER_TRAIN
        logger = logging.getLogger(__name__)
        logger.info("Using training sampler {}".format(sampler_name))
        
        sampler = TrainingSampler(len(dataset))

        return build_batch_data_loader(
            dataset,
            sampler,
            cfg.SOLVER.IMS_PER_BATCH,
            aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )
        
    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        from detectron2.modeling import build_distill_model, build_model
        if cfg.IOD.DISTILL:
            model = build_distill_model(cfg)
        else:
            model = build_model(cfg)
        logger = logging.getLogger(__name__)
        logger.info("Model:\n{}".format(model))
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return PascalVOCDetectionEvaluator(dataset_name, cfg.IOD.OLD_CLS, cfg.IOD.NEW_CLS) 

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
        if comm.is_main_process(): verify_results(cfg, res)
        return res
    
    trainer = Trainer(cfg)
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