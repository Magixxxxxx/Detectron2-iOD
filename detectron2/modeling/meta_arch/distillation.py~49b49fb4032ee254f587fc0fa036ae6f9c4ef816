import torch
import torch.distributed as dist
from torch import nn
import numpy as np

def calculate_rpn_distillation_loss(rpn_output_source, rpn_output_target, cls_loss='filtered_l2', bbox_loss='l2', bbox_threshold=0.1):

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
    # l2_loss = nn.MSELoss(size_average=False, reduce=False)
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


def calculate_feature_distillation_loss(source_features, target_features, loss='normalized_filtered_l1'):  # pixel-wise

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


from torch.nn import functional as F

def calculate_roi_distillation_loss(target_scores, target_bboxes, soften_scores, soften_bboxes, cls_preprocess='normalization', cls_loss='l2', bbs_loss='l2', temperature=1):
    '''
    64*80
    64*21
    '''
    # soften_scores, soften_bboxes = soften_results
    # # images = to_image_list(images)
    # # features, backbone_features = self.backbone(images.tensors)  # extra image features from backbone network
    # target_scores, target_bboxes = self.roi_heads.calculate_soften_label(features, soften_proposals, soften_results)

    num_of_distillation_categories = 15
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