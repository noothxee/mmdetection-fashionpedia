# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer


@HEADS.register_module()
class AttributeHead(BaseModule):

    def __init__(self,
                 with_avg_pool=False,
                 with_atr=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_attributes=294,
                 atr_predictor_cfg=dict(type='Linear'),
                 loss_atr=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 init_cfg=False):

        super(AttributeHead, self).__init__(init_cfg)
        assert with_atr
        self.with_avg_pool = with_avg_pool
        self.with_atr = with_atr
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_attributes = num_attributes
        self.atr_predictor_cfg = atr_predictor_cfg
        self.fp16_enabled = False

        self.loss_atr = build_loss(loss_atr)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels = self.roi_feat_area

        if self.with_atr:
            if self.custom_atr_channels:
                atr_channels = self.loss_atr.get_atr_channels(
                    self.num_attributes)
            else:
                atr_channels = num_attributes + 1
            self.fc_atr = build_linear_layer(
                self.atr_predictor_cfg,
                in_features=in_channels,
                out_features=atr_channels)

        if init_cfg is None:
            self.init_cfg = []
            if self.with_atr:
                self.init_cfg += [
                    dict(
                        type='Normal', std=0.01, override=dict(name='fc_atr')
                    )
                ]

    @property
    def custom_atr_channels(self):
        return getattr(self.loss_atr, 'custom_atr_channels', False)

    @property
    def custom_activation(self):
        return getattr(self.loss_atr, 'custom_activation', False)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))
        atr_score = self.fc_atr(x) if self.with_atr else None
        return atr_score
    
    def _get_target_single(self, pos_bboxes, neg_bboxes, pos_gt_bboxes,
                           pos_gt_attributes, cfg):

        num_pos = pos_bboxes.size(0)
        num_neg = neg_bboxes.size(0)
        num_samples = num_pos + num_neg
        
        import os.path as osp
        import time
        from mmdet.utils import collect_env, get_root_logger
        
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = osp.join('/home/noothxee/th_dev/mmdetection-fashionpedia/work_dirs/mask_rcnn_r50_fpn_1x_coco', f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level='INFO')
        
        
        attributes = pos_bboxes.new_full((num_samples, ),
                                     self.num_attributes,
                                     dtype=torch.long)
        attribute_weights = pos_bboxes.new_zeros(num_samples)

        if num_pos > 0:
            attributes[:pos_gt_attributes.size(dim=0)] = pos_gt_attributes
            pos_weight = 1.0 if cfg.pos_weight <= 0 else cfg.pos_weight
            attribute_weights[:pos_gt_attributes.size(dim=0)] = pos_weight
        if num_neg > 0:
            attribute_weights[-num_neg:] = 1.0

        return attributes, attribute_weights

    def get_targets(self,
                    sampling_results,
                    gt_attributes,
                    rcnn_train_cfg,
                    concat=True):
        
        pos_bboxes_list = [res.pos_bboxes for res in sampling_results]
        neg_bboxes_list = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes_list = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_attributes_list = [res.pos_gt_attributes for res in sampling_results]
        
        attributes, attribute_weights= multi_apply(
            self._get_target_single,
            pos_bboxes_list,
            neg_bboxes_list,
            pos_gt_bboxes_list,
            pos_gt_attributes_list,
            cfg=rcnn_train_cfg)

        if concat:
            attributes = torch.cat(attributes, 0)
            attribute_weights = torch.cat(attribute_weights, 0)
            
        return attributes, attribute_weights

    @force_fp32(apply_to=('atr_score'))
    def loss(self,
             atr_score,
             attributes,
             attribute_weights,
             reduction_override=None):

        losses = dict()

        if atr_score is not None:
            if atr_score.numel() > 0:
                loss_atr_ = self.loss_atr(
                    atr_score,
                    attributes,
                    attribute_weights,
                    reduction_override=reduction_override)
                if isinstance(loss_atr_, dict):
                    losses.update(loss_atrs_)
                else:
                    losses['loss_atr'] = loss_atr_
                if self.custom_activation:
                    acc_ = self.loss_atr.get_accuracy(atr_score, attributes)
                    losses.update(acc_)
                else:
                    losses['atr_acc'] = accuracy(atr_score, attributes)
        return losses
