# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .atr_head import AttributeHead


@HEADS.register_module()
class ConvFCAtrHead(AttributeHead):
    r"""
        atr_conv -> atr_fcs -> 
    """  # noqa: W605

    def __init__(self,
                 num_atr_convs=0,
                 num_atr_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCAtrHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        assert (num_atr_convs + num_atr_fcs > 0)
        if not self.with_atr:
            assert num_atr_convs == 0 and num_atr_fcs == 0

        self.num_atr_convs = num_atr_convs
        self.num_atr_fcs = num_atr_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add atr specific branch
        self.atr_convs, self.atr_fcs, self.atr_last_dim = \
            self._add_conv_fc_branch(
                self.num_cls_convs, self.num_cls_fcs, self.shared_out_channels)

        if not self.with_avg_pool:
            if self.num_atr_fcs == 0:
                self.atr_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)
        # reconstruct fc_cls and fc_reg since input channels are changed
        if self.with_atr:
            if self.custom_atr_channels:
                atr_channels = self.loss_atr.get_atr_channels(
                    self.num_attributes)
            else:
                atr_channels = self.num_attributes
            self.fc_atr = build_linear_layer(
                self.atr_predictor_cfg,
                in_features=self.atr_last_dim,
                out_features=atr_channels)

        if init_cfg is None:
            self.init_cfg += [
                dict(
                    type='Xavier',
                    distribution='uniform',
                    override=[
                        dict(name='atr_fcs'),
                    ])
            ]

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch.

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def forward(self, x):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)

            x = x.flatten(1)

            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_atr = x

        for conv in self.atr_convs:
            x_atr = conv(x_atr)
        if x_atr.dim() > 2:
            if self.with_avg_pool:
                x_atr = self.avg_pool(x_atr)
            x_atr = x_atr.flatten(1)
        for fc in self.atr_fcs:
            x_atr = self.relu(fc(x_atr))

        atr_score = self.fc_atr(x_atr) if self.with_atr else None
        return atr_score


@HEADS.register_module()
class Shared2FCAtrHead(ConvFCAtrHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared2FCAtrHead, self).__init__(
            num_atr_convs=2,
            num_atr_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)


@HEADS.register_module()
class Shared4Conv1FCAtrHead(ConvFCAtrHead):

    def __init__(self, fc_out_channels=1024, *args, **kwargs):
        super(Shared4Conv1FCAtrHead, self).__init__(
            num_atr_convs=4,
            num_atr_fcs=1,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
