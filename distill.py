import copy
import os
import logging
from collections import OrderedDict
from random import randint
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
import torch, sys, logging, time

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

class Trainer(DefaultTrainer):

    def __init__(self, cfg):
        super().__init__(cfg)
        
        if comm.get_world_size() > 1:
            md = self.model.module
        if cfg.IOD.DISTILL:
            sd = torch.load(cfg.MODEL.WEIGHTS, map_location='cpu')['model']  # why map_location leads to cruption
            md.load_state_dict(sd, strict = False)
            md.t_model.load_state_dict(sd)        
        else:
            self.resume_or_load()

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
        data_time = time.perf_counter() - start

        loss_dict = self.model(data)
        # loss_dict['loss_cls'] *= 2
        # loss_dict['loss_box_reg'] *= 2
        # loss_dict['loss_rpn_cls'] *= 2
        # loss_dict['loss_rpn_loc'] *= 2

        # loss_dict['dist_rpn_loss'] *= 2
        # loss_dict['dist_feature_loss'] *= 2
        # loss_dict['distill_rcn_loss'] *= 2
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
    def build_train_loader(cls, cfg):
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
        # valid_classes = range(cfg.IOD.OLD_CLS, cfg.IOD.OLD_CLS + cfg.IOD.NEW_CLS)
        valid_classes = range(cfg.IOD.OLD_CLS)
        dataset_dicts = filter_classes_instances(dataset_dicts, valid_classes)
        print(len(dataset_dicts))

        if cfg.IOD.MEMORY:
            dataset_dicts += get_detection_dataset_dicts(
                cfg.IOD.MEMORY,
                filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
                min_keypoints=0,
                proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None
                )
        
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