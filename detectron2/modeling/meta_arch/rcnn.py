# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from matplotlib import image
import numpy as np
from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from detectron2.config import configurable
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from ..backbone import Backbone, build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads
from .build import META_ARCH_REGISTRY

from .build import build_model

__all__ = ["GeneralizedRCNN", "ProposalNetwork", "DistillRCNN"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses, rpn_logits, rpn_boxes = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
            
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _, _, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    #ZJW
    def get_distill_target(self, batched_inputs):

        distill_target = {}

        images = self.preprocess_image(batched_inputs)

        features = self.backbone(images.tensor)
        distill_target['t_features'] = features

        pred_objectness_logits, pred_anchor_deltas, proposals = self.proposal_generator.get_distill_target(images, features)
        distill_target['t_rpn_logits'] = pred_objectness_logits
        distill_target['t_rpn_boxes'] = pred_anchor_deltas

        rcn_input_proposals, rcn_cls, rcn_reg = self.roi_heads.get_distill_target(features, proposals)
        distill_target['rcn_input_proposals'] = rcn_input_proposals
        distill_target['t_rcn_cls'] = rcn_cls
        distill_target['t_rcn_reg'] = rcn_reg

        del images, features, pred_objectness_logits, pred_anchor_deltas, proposals, rcn_input_proposals, rcn_cls, rcn_reg
        torch.cuda.empty_cache()
        return distill_target


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    """
    A meta architecture that only predicts object proposals.
    """

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results


@META_ARCH_REGISTRY.register()
class DistillRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        t_model,
        hp
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element,
                representing the per-channel mean and std to be used to normalize
                the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

        self.t_model = t_model
        self.hp = hp

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        t_model = build_model(cfg)
        t_model.requires_grad_(False)

        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "t_model": t_model,
            "hp": {"old_cls": cfg.IOD.OLD_CLS, "new_cls": cfg.IOD.OLD_CLS}
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if self.proposal_generator:
            proposals, proposal_losses, rpn_logits, rpn_boxes = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, proposals)
            
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # ZJW
        with torch.no_grad(): dt = self.t_model.get_distill_target(batched_inputs)

        # rpn
        # distill_rpn_losses = self.rpn_distill_losses(dt['t_rpn_logits'][0], dt['t_rpn_boxes'][0], rpn_logits[0], rpn_boxes[0])
        # distill_rpn_losses = self.calculate_rpn_distillation_loss((rpn_logits, rpn_boxes), (dt['t_rpn_logits'], dt['t_rpn_boxes']))
        # losses.update(distill_rpn_losses)

        # feature
        # distill_feature_losses = self.feature_distill_losses(dt['t_features'], features)
        distill_feature_losses = self.calculate_feature_distillation_loss(features, dt['t_features'])
        losses.update(distill_feature_losses)

        # rcn
        box_features = self.roi_heads._shared_roi_transform(
            [features["res4"]], dt['rcn_input_proposals']
        )
        rcn_cls, rcn_reg = self.roi_heads.box_predictor(box_features.mean(dim=[2, 3]))

        # distill_rcn_losses = self.rcn_distill_losses(dt['t_rcn_cls'], dt['t_rcn_reg'], rcn_cls, rcn_reg)
        distill_rcn_losses = self.calculate_roi_distillation_loss(dt['t_rcn_cls'], dt['t_rcn_reg'], rcn_cls, rcn_reg )
        losses.update(distill_rcn_losses)

        del dt, rcn_cls, rcn_reg, box_features, features, images
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _, _, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    #ZJW
    def calculate_rpn_distillation_loss(self, rpn_output_source, rpn_output_target, cls_loss='filtered_l2', bbox_loss='l2', bbox_threshold=0.1):

        rpn_objectness_source, rpn_bbox_regression_source = rpn_output_source
        rpn_objectness_target, rpn_bbox_regression_target = rpn_output_target

        # calculate rpn classification loss
        num_source_rpn_objectness = len(rpn_objectness_source)
        num_target_rpn_objectness = len(rpn_objectness_target)
        final_rpn_cls_distillation_loss = []
        objectness_difference = []

        if num_source_rpn_objectness == num_target_rpn_objectness:
            for i in range(num_target_rpn_objectness):
                current_source_rpn_objectness = rpn_objectness_source[i]
                current_target_rpn_objectness = rpn_objectness_target[i]
                if cls_loss == 'filtered_l1':
                    rpn_objectness_difference = current_source_rpn_objectness - current_target_rpn_objectness
                    objectness_difference.append(rpn_objectness_difference)
                    filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                    rpn_distillation_loss = torch.max(rpn_objectness_difference, filter)
                    final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                    del filter
                    torch.cuda.empty_cache()  # Release unoccupied memory
                elif cls_loss == 'filtered_l2':
                    rpn_objectness_difference = current_source_rpn_objectness - current_target_rpn_objectness
                    objectness_difference.append(rpn_objectness_difference)
                    filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                    rpn_difference = torch.max(rpn_objectness_difference, filter)
                    rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                    final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                    del filter
                    torch.cuda.empty_cache()  # Release unoccupied memory
                elif cls_loss == 'normalized_filtered_l2':
                    avrage_source_rpn_objectness = torch.mean(current_source_rpn_objectness)
                    average_target_rpn_objectness = torch.mean(current_target_rpn_objectness)
                    normalized_source_rpn_objectness = current_source_rpn_objectness - avrage_source_rpn_objectness
                    normalized_target_rpn_objectness = current_target_rpn_objectness - average_target_rpn_objectness
                    rpn_objectness_difference = normalized_source_rpn_objectness - normalized_target_rpn_objectness
                    objectness_difference.append(rpn_objectness_difference)
                    filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                    rpn_difference = torch.max(rpn_objectness_difference, filter)
                    rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                    final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                    del filter
                    torch.cuda.empty_cache()  # Release unoccupied memory
                elif cls_loss == 'masked_filtered_l2':
                    source_mask = current_source_rpn_objectness.clone()
                    source_mask[current_source_rpn_objectness >= 0.7] = 1  # rpn threshold for foreground
                    source_mask[current_source_rpn_objectness < 0.7] = 0
                    rpn_objectness_difference = current_source_rpn_objectness - current_target_rpn_objectness
                    masked_rpn_objectness_difference = rpn_objectness_difference * source_mask
                    objectness_difference.append(masked_rpn_objectness_difference)
                    filter = torch.zeros(current_source_rpn_objectness.size()).to('cuda')
                    rpn_difference = torch.max(masked_rpn_objectness_difference, filter)
                    rpn_distillation_loss = torch.mul(rpn_difference, rpn_difference)
                    final_rpn_cls_distillation_loss.append(torch.mean(rpn_distillation_loss))
                    del filter
                    torch.cuda.empty_cache()  # Release unoccupied memory
                else:
                    raise ValueError("Wrong loss function for rpn classification distillation")
        else:
            raise ValueError("Wrong rpn objectness output")
        final_rpn_cls_distillation_loss = sum(final_rpn_cls_distillation_loss)/num_source_rpn_objectness

        # calculate rpn bounding box regression loss
        num_source_rpn_bbox = len(rpn_bbox_regression_source)
        num_target_rpn_bbox = len(rpn_bbox_regression_target)
        final_rpn_bbs_distillation_loss = []

        l2_loss = nn.MSELoss(reduction='mean')

        if num_source_rpn_bbox == num_target_rpn_bbox:
            for i in range(num_target_rpn_bbox):
                current_source_rpn_bbox = rpn_bbox_regression_source[i]
                current_target_rpn_bbox = rpn_bbox_regression_target[i]
                current_objectness_difference = objectness_difference[i]
                current_objectness_mask = current_objectness_difference.clone()

                current_objectness_mask = current_objectness_mask.unsqueeze(dim = 2)

                current_objectness_mask[current_objectness_difference > bbox_threshold] = 1
                current_objectness_mask[current_objectness_difference <= bbox_threshold] = 0
                masked_source_rpn_bbox = current_source_rpn_bbox * current_objectness_mask
                masked_target_rpn_bbox = current_target_rpn_bbox * current_objectness_mask
                if bbox_loss == 'l2':
                    current_bbox_distillation_loss = l2_loss(masked_source_rpn_bbox, masked_target_rpn_bbox)
                    final_rpn_bbs_distillation_loss.append(current_bbox_distillation_loss)
                elif bbox_loss == 'l1':
                    current_bbox_distillation_loss = torch.abs(masked_source_rpn_bbox - masked_source_rpn_bbox)
                    final_rpn_bbs_distillation_loss.append(torch.mean(torch.mean(torch.sum(current_bbox_distillation_loss, dim=2), dim=1), dim=0))
                elif bbox_loss == 'None':
                    final_rpn_bbs_distillation_loss.append(0)
                else:
                    raise ValueError('Wrong loss function for rpn bounding box regression distillation')
        else:
            raise ValueError('Wrong RPN bounding box regression output')
        final_rpn_bbs_distillation_loss = sum(final_rpn_bbs_distillation_loss)/num_source_rpn_bbox

        final_rpn_loss = final_rpn_cls_distillation_loss + final_rpn_bbs_distillation_loss
        final_rpn_loss.to('cuda')
        del rpn_output_source, rpn_output_target
        torch.cuda.empty_cache()
        return {'distill_rpn_loss':final_rpn_loss}
 
    def calculate_feature_distillation_loss(self, source_features, target_features, loss='normalized_filtered_l1'):  # pixel-wise

        num_source_features = len(source_features)
        num_target_fetures = len(target_features)
        final_feature_distillation_loss = []

        if num_source_features == num_target_fetures:
            for i in source_features.keys():
                source_feature = source_features[i]
                target_feature = target_features[i]
                if loss == 'l2':
                    # l2_loss = nn.MSELoss(size_average=False, reduce=False)
                    l2_loss = nn.MSELoss(reduction='mean')
                    feature_distillation_loss = l2_loss(source_feature, target_feature)
                    final_feature_distillation_loss.append(feature_distillation_loss)
                elif loss == 'l1':
                    feature_distillation_loss = torch.abs(source_feature - target_feature)
                    final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
                elif loss == 'normalized_filtered_l1':
                    source_feature_avg = torch.mean(source_feature)
                    target_feature_avg = torch.mean(target_feature)
                    normalized_source_feature = source_feature - source_feature_avg  # normalize features
                    normalized_target_feature = target_feature - target_feature_avg
                    feature_difference = normalized_source_feature - normalized_target_feature
                    feature_size = feature_difference.size()
                    filter = torch.zeros(feature_size).to('cuda')
                    feature_distillation_loss = torch.max(feature_difference, filter)
                    final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
                    del filter
                    torch.cuda.empty_cache()  # Release unoccupied memory
                elif loss == 'normalized_filtered_l2':
                    source_feature_avg = torch.mean(source_feature)
                    target_feature_avg = torch.mean(target_feature)
                    normalized_source_feature = source_feature - source_feature_avg  # normalize features
                    normalized_target_feature = target_feature - target_feature_avg  # normalize features
                    feature_difference = normalized_source_feature - normalized_target_feature
                    feature_size = feature_difference.size()
                    filter = torch.zeros(feature_size).to('cuda')
                    feature_distillation = torch.max(feature_difference, filter)
                    feature_distillation_loss = torch.mul(feature_distillation, feature_distillation)
                    final_feature_distillation_loss.append(torch.mean(feature_distillation_loss))
                    del filter
                    torch.cuda.empty_cache()  # Release unoccupied memory
                else:
                    raise ValueError("Wrong loss function for feature distillation")
        else:
            raise ValueError("Number of source features must equal to number of target features")

        final_feature_distillation_loss = sum(final_feature_distillation_loss)
        del source_features, target_features
        torch.cuda.empty_cache()
        return {'distill_feature_loss':final_feature_distillation_loss}

    def calculate_roi_distillation_loss(
        self, target_scores, target_bboxes, soften_scores, soften_bboxes, 
        cls_preprocess='normalization', cls_loss='l2', bbs_loss='l2', temperature=1):
        '''
        64*80
        64*21
        '''
        # soften_scores, soften_bboxes = soften_results
        # # images = to_image_list(images)
        # # features, backbone_features = self.backbone(images.tensors)  # extra image features from backbone network
        # target_scores, target_bboxes = self.roi_heads.calculate_soften_label(features, soften_proposals, soften_results)

        num_of_distillation_categories = self.hp['old_cls']
        # compute distillation loss
        if cls_preprocess == 'sigmoid':
            soften_scores = F.sigmoid(soften_scores)
            target_scores = F.sigmoid(target_scores)
            modified_soften_scores = soften_scores[:, : num_of_distillation_categories]  # include background
            modified_target_scores = target_scores[:, : num_of_distillation_categories]  # include background
        elif cls_preprocess == 'softmax':  # exp(x_i) / exp(x).sum()
            soften_scores = F.softmax(soften_scores)
            target_scores = F.softmax(target_scores)
            modified_soften_scores = soften_scores[:, : num_of_distillation_categories]  # include background
            modified_target_scores = target_scores[:, : num_of_distillation_categories]  # include background
        elif cls_preprocess == 'log_softmax':  # log( exp(x_i) / exp(x).sum() )
            soften_scores = F.log_softmax(soften_scores)
            target_scores = F.log_softmax(target_scores)
            modified_soften_scores = soften_scores[:, : num_of_distillation_categories]  # include background
            modified_target_scores = target_scores[:, : num_of_distillation_categories]  # include background
        elif cls_preprocess == 'normalization':
            class_wise_soften_scores_avg = torch.mean(soften_scores, dim=1).view(-1, 1)
            class_wise_target_scores_avg = torch.mean(target_scores, dim=1).view(-1, 1)
            normalized_soften_scores = torch.sub(soften_scores, class_wise_soften_scores_avg)
            normalized_target_scores = torch.sub(target_scores, class_wise_target_scores_avg)
            modified_soften_scores = normalized_target_scores[:, : num_of_distillation_categories]  # include background
            modified_target_scores = normalized_soften_scores[:, : num_of_distillation_categories]  # include background
        elif cls_preprocess == 'raw':
            modified_soften_scores = soften_scores[:, : num_of_distillation_categories]  # include background
            modified_target_scores = target_scores[:, : num_of_distillation_categories]  # include background
        else:
            raise ValueError("Wrong preprocessing method for raw classification output")

        if cls_loss == 'l2':
            l2_loss = nn.MSELoss(reduction='mean')
            class_distillation_loss = l2_loss(modified_soften_scores, modified_target_scores)
            # class_distillation_loss = torch.mean(torch.mean(class_distillation_loss, dim=1), dim=0)  # average towards categories and proposals
        elif cls_loss == 'cross-entropy':  # softmax/sigmoid + cross-entropy
            class_distillation_loss = - modified_soften_scores * torch.log(modified_target_scores)
            class_distillation_loss = torch.mean(torch.mean(class_distillation_loss, dim=1), dim=0)  # average towards categories and proposals
        elif cls_loss == 'softmax cross-entropy with temperature':  # raw + softmax cross-entropy with temperature
            log_softmax = nn.LogSoftmax()
            softmax = nn.Softmax()
            class_distillation_loss = - softmax(modified_soften_scores/temperature) * log_softmax(modified_target_scores/temperature)
            class_distillation_loss = class_distillation_loss * temperature * temperature
            class_distillation_loss = torch.mean(torch.mean(class_distillation_loss, dim=1), dim=0)  # average towards categories and proposals
        elif cls_loss == 'filtered_l2':
            cls_difference = modified_soften_scores - modified_target_scores
            filter = torch.zeros(modified_soften_scores.size()).to('cuda')
            class_distillation_loss = torch.max(cls_difference, filter)
            class_distillation_loss = class_distillation_loss * class_distillation_loss
            class_distillation_loss = torch.mean(torch.mean(class_distillation_loss, dim=1), dim=0)  # average towards categories and proposals
            del filter
            torch.cuda.empty_cache()  # Release unoccupied memory
        else:
            raise ValueError("Wrong loss function for classification")

        # compute distillation bbox loss
        # modified_soften_boxes = soften_bboxes[:, 1:, :]  # exclude background bbox
        # modified_target_bboxes = target_bboxes[:, 1:num_of_distillation_categories, :]  # exclude background bbox

        modified_soften_boxes = soften_bboxes[:, :num_of_distillation_categories*4]  # exclude background bb:num_of_distillation_categoriesox
        modified_target_bboxes = target_bboxes[:, :num_of_distillation_categories*4]  # exclude background bbox

        if bbs_loss == 'l2':
            l2_loss = nn.MSELoss(reduction='mean')
            bbox_distillation_loss = l2_loss(modified_target_bboxes, modified_soften_boxes)
            # bbox_distillation_loss = torch.mean(torch.mean(torch.sum(bbox_distillation_loss, dim=2), dim=1), dim=0)  # average towards categories and proposals
        # elif bbs_loss == 'smooth_l1':
        #     num_bboxes = modified_target_bboxes.size()[0]
        #     num_categories = modified_target_bboxes.size()[1]
        #     bbox_distillation_loss = smooth_l1_loss(modified_target_bboxes, modified_soften_boxes, size_average=False, beta=1)
        #     bbox_distillation_loss = bbox_distillation_loss / (num_bboxes * num_categories)  # average towards categories and proposals
        else:
            raise ValueError("Wrong loss function for bounding box regression")

        roi_distillation_losses = torch.add(class_distillation_loss, bbox_distillation_loss)
        del target_scores, target_bboxes, soften_scores, soften_bboxes
        torch.cuda.empty_cache()
        return {"distill_rcn_loss": roi_distillation_losses}

    def rpn_distill_losses(self, t_rpn_logits, t_rpn_proposals, rpn_logits, rpn_proposals):

        loss = {}
        t = 0.1

        #logits loss
        filter_logits = torch.zeros(rpn_logits.shape).to('cuda')
        diff_logits = torch.max(t_rpn_logits - rpn_logits, filter_logits) 
        logits_loss = torch.mean(torch.mul(diff_logits, diff_logits)) 

        #box loss
        mask_box = t_rpn_logits.clone()
        mask_box[t_rpn_logits > rpn_logits + t] = 1
        mask_box[t_rpn_logits <= rpn_logits + t] = 0
        mask_box = mask_box.unsqueeze(dim = 2)

        diff_boxes = torch.mul(t_rpn_proposals - rpn_proposals, mask_box)
        box_loss = torch.mean(torch.mul(diff_boxes, diff_boxes))

        loss['dist_rpn_loss'] = torch.add(box_loss, logits_loss)
        return loss

    def rcn_distill_losses(self, t_rcn_cls, t_rcn_reg, rcn_cls, rcn_reg):
        '''
        t_rcn_cls: tensor(rand_k * 21)
        '''
        loss = {}
        loss_func = nn.MSELoss(reduction='mean')

        #logits loss
        t_rcn_cls = nn.functional.softmax(t_rcn_cls[:, :15*4], dim = 1)
        rcn_cls = nn.functional.softmax(rcn_cls[:, :15*4], dim = 1)
        logits_loss = loss_func(t_rcn_cls, rcn_cls)

        #box loss
        box_loss = loss_func(t_rcn_reg, rcn_reg)
        loss['dist_rcn_loss'] = torch.add(box_loss, logits_loss)

        del t_rcn_cls, t_rcn_reg, rcn_cls, rcn_reg
        return loss

    def feature_distill_losses(self, t_feature, feature):
        loss = {}
        t_feature = t_feature['res4'] 
        feature = feature['res4']
        filter_feature = torch.zeros(t_feature.shape).to('cuda')
        feature_distillation_loss = torch.max(t_feature - feature, filter_feature)
        loss['dist_feature_loss'] = torch.mean(feature_distillation_loss)

        del t_feature, feature, filter_feature
        return loss
