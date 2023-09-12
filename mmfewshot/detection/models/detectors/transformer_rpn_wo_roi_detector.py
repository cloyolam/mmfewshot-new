# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional

import copy
import numpy as np
import torch
from mmcv.runner import auto_fp16
from mmcv.utils import ConfigDict
from mmdet.core import bbox2roi
from mmdet.models.builder import DETECTORS
from torch import Tensor

from .query_support_detector import QuerySupportDetector


@DETECTORS.register_module()
class TransformerRPNWoRoiHeadDetector(QuerySupportDetector):
    """Implementation of Cross Attention Transformer: https://arxiv.org/abs/2104.14984

    Args:
        backbone (dict): Config of the backbone for query data.
        neck (dict | None): Config of the neck for query data and
            probably for support data. Default: None.
        support_backbone (dict | None): Config of the backbone for
            support data only. If None, support and query data will
            share same backbone. Default: None.
        support_neck (dict | None): Config of the neck for support
            data only. Default: None.
        rpn_head (dict | None): Config of rpn_head. Default: None.
        roi_head (dict | None): Config of roi_head. Default: None.
        train_cfg (dict | None): Training config. Useless in CenterNet,
            but we keep this variable for SingleStageDetector. Default: None.
        test_cfg (dict | None): Testing config of CenterNet. Default: None.
        pretrained (str | None): model pretrained path. Default: None.
        init_cfg (dict | list[dict] | None): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 backbone: ConfigDict,
                 neck: Optional[ConfigDict] = None,
                 support_backbone: Optional[ConfigDict] = None,
                 support_neck: Optional[ConfigDict] = None,
                 rpn_head: Optional[ConfigDict] = None,
                 roi_head: Optional[ConfigDict] = None,
                 train_cfg: Optional[ConfigDict] = None,
                 test_cfg: Optional[ConfigDict] = None,
                 pretrained: Optional[ConfigDict] = None,
                 init_cfg: Optional[ConfigDict] = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            support_backbone=support_backbone,
            support_neck=support_neck,
            rpn_head=rpn_head,
            # roi_head=roi_head,
            roi_head=None,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.is_model_init = False
        # save support template features for model initialization,
        # `_forward_saved_support_dict` used in :func:`forward_model_init`.
        self._forward_saved_support_dict = {
            'gt_labels': [],
            'res4_roi_feats': [],
            # 'res5_roi_feats': []
        }
        # save processed support template features for inference,
        # the processed support template features are generated
        # in :func:`model_init`
        self.inference_support_dict = {}

    @auto_fp16(apply_to=('img', ))
    def extract_support_feat(self, img: Tensor) -> List[Tensor]:
        """Extract features of support data.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            list[Tensor]: Features of support images, each item with shape
                 (N, C, H, W).
        """
        feats = self.support_backbone(img)
        # Comment this, neck is applied to query and support at the same time
        # if self.support_neck is not None:
        #     feats = self.support_neck(feats)
        return feats

    def forward_model_init(self,
                           img: Tensor,
                           img_metas: List[Dict],
                           gt_bboxes: List[Tensor] = None,
                           gt_labels: List[Tensor] = None,
                           **kwargs) -> Dict:
        """Extract and save support features for model initialization.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box.

        Returns:
            dict: A dict contains following keys:

                - `gt_labels` (Tensor): class indices corresponding to each
                    feature.
                - `res4_roi_feat` (Tensor): roi features of res4 layer.
                - `res5_roi_feat` (Tensor): roi features of res5 layer.
        """
        self.is_model_init = False
        # extract support template features will reset `is_model_init` flag
        assert gt_bboxes is not None and gt_labels is not None, \
            'forward support template require gt_bboxes and gt_labels.'
        assert len(gt_labels) == img.size(0), \
            'Support instance have more than two labels'

        feats = self.extract_support_feat(img)
        rois = bbox2roi([bboxes for bboxes in gt_bboxes])
        res4_roi_feat = self.rpn_head.extract_roi_feat(feats, rois)
        # res5_roi_feat = self.roi_head.extract_roi_feat(feats, rois)
        self._forward_saved_support_dict['gt_labels'].extend(gt_labels)
        self._forward_saved_support_dict['res4_roi_feats'].append(
            res4_roi_feat)
        # self._forward_saved_support_dict['res5_roi_feats'].append(
        #     res5_roi_feat)

        return {
            'gt_labels': gt_labels,
            'res4_roi_feats': res4_roi_feat,
            # 'res5_roi_feats': res5_roi_feat
        }

    def model_init(self) -> None:
        # print("Entering model_init in TransformerRPNWoRoiHeadDetector...")
        """process the saved support features for model initialization."""
        self.inference_support_dict.clear()
        gt_labels = torch.cat(self._forward_saved_support_dict['gt_labels'])
        # print(f"  gt_labels.size() = {gt_labels.size()}")
        # used for attention rpn head
        res4_roi_feats = torch.cat(
            self._forward_saved_support_dict['res4_roi_feats'])
        # print(f"  res4_roi_feats.size() = {res4_roi_feats.size()}")
        # used for multi relation head
        # res5_roi_feats = torch.cat(
        #     self._forward_saved_support_dict['res5_roi_feats'])
        class_ids = set(gt_labels.data.tolist())
        # print(f"  class_ids = {class_ids}")
        for class_id in class_ids:
            # print(f"  class_id: {class_id}")
            # print(f"    before_averaging: {res4_roi_feats[gt_labels == class_id].size()}")
            # don't apply GAP when using CAT block
            # res4_roi_feats -> [1, C, roi_height, roi_width] -> [1, 1024, 14, 14]
            self.inference_support_dict[class_id] = {
                'res4_roi_feats':
                res4_roi_feats[gt_labels == class_id].mean([0], True),
                # 'res5_roi_feats':
                # res5_roi_feats[gt_labels == class_id].mean([0], True)
            }
            # print(f"    after averaging: {self.inference_support_dict[class_id]['res4_roi_feats'].size()}")
        # set the init flag
        self.is_model_init = True
        # clear support dict
        for k in self._forward_saved_support_dict.keys():
            self._forward_saved_support_dict[k].clear()

    def simple_test(self,
                    img: Tensor,
                    img_metas: List[Dict],
                    proposals: Optional[List[Tensor]] = None,
                    rescale: bool = False) -> List[List[np.ndarray]]:
        """Test without augmentation.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            proposals (list[Tensor] | None): override rpn proposals with
                custom proposals. Use when `with_rpn` is False. Default: None.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        # print("Entering simple_test in TransformerRPNWoROIDetector...")
        # assert self.with_bbox, 'Bbox head must be implemented.'
        assert len(img_metas) == 1, 'Only support single image inference.'
        if (self.inference_support_dict == {}) or (not self.is_model_init):
            # process the saved support features
            self.model_init()

        results_dict = {}
        query_feats = self.extract_feat(img)
        # print(f"  len(query_feats) = {len(query_feats)}")
        # print(f"  query_feats[0].size() = {query_feats[0].size()}")
        for class_id in self.inference_support_dict.keys():
            support_res4_roi_feat = \
                self.inference_support_dict[class_id]['res4_roi_feats']
            # support_res5_roi_feat = \
            #     self.inference_support_dict[class_id]['res5_roi_feats']
            # print(f"  support_res4_roi_feat.size() = {support_res4_roi_feat.size()}")

            # Apply CAT between query and support features
            if self.neck is not None:
                # print("Applying CAT between query and support features...")
                query_feats = query_feats[0]
                # print("Shapes before CAT neck:")
                # print(f"  query_feats.size() = {query_feats.size()}")
                # print(f"  support_res4_roi_feat.size() = {support_res4_roi_feat.size()}")
                query_feats, _ = self.neck(query_feats, support_res4_roi_feat)
                query_feats = [query_feats]
            # print("After CAT neck:")
            # print(f"  len(query_feats) = {len(query_feats)}")
            # print(f"  query_feats[0].size() = {query_feats[0].size()}")

            if proposals is None:
                proposal_list = self.rpn_head.simple_test(query_feats,
                                                          img_metas,
                                                          rescale=True)
            else:
                proposal_list = proposals
            results_dict[class_id] = proposal_list[0].detach().cpu().numpy()

        '''
            results_dict[class_id] = self.roi_head.simple_test(
                query_feats,
                support_res5_roi_feat,
                proposal_list,
                img_metas,
                rescale=rescale)
        '''
        results = [
            results_dict[i] for i in sorted(results_dict.keys())
            if len(results_dict[i])
        ]
        return [results]

    # Override QuerySupportDetector methods

    @auto_fp16(apply_to=('img', ))
    def extract_query_feat(self, img: Tensor) -> List[Tensor]:
        """Extract features of query data.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            list[Tensor]: Features of support images, each item with shape
                 (N, C, H, W).
        """
        feats = self.backbone(img)
        # Comment this, neck is applied to query and support at the same time
        # if self.with_neck:
        #     feats = self.neck(feats)
        return feats

    def extract_feat(self, img: Tensor) -> List[Tensor]:
        """Extract features of query data.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            list[Tensor]: Features of query images.
        """
        return self.extract_query_feat(img)

    def forward_train(self,
                      query_data: Dict,
                      support_data: Dict,
                      proposals: Optional[List] = None,
                      **kwargs) -> Dict:
        """Forward function for training.

        Args:
            query_data (dict): In most cases, dict of query data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            support_data (dict):  In most cases, dict of support data contains:
                `img`, `img_metas`, `gt_bboxes`, `gt_labels`,
                `gt_bboxes_ignore`.
            proposals (list): Override rpn proposals with custom proposals.
                Use when `with_rpn` is False. Default: None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # print("Entering forward_train in TransformerRPNDetector ...")
        query_img = query_data['img']
        support_img = support_data['img']

        '''
        print("Shapes before backbone...")
        print(f"  query_img.shape = {query_img.shape}")
        print(f"  support_img.shape = {support_img.shape}")
        '''

        query_feats = self.extract_query_feat(query_img)
        support_feats = self.extract_support_feat(support_img)

        '''
        print(f"  type(query_feats) = {type(query_feats)}")
        print(f"  type(support_feats) = {type(support_feats)}")
        print(f"  len(query_feats) = {len(query_feats)}")
        print(f"  len(support_feats) = {len(support_feats)}")

        print("Shapes after backbone:")
        print(f"  query_feats[0].shape = {query_feats[0].shape}")
        print(f"  support_feats[0].shape = {support_feats[0].shape}")
        '''

        assert len(query_feats) == len(support_feats) == 1, \
            'len(query_feats) or len(support_feats) is different from 1'

        # Apply block of Cross Attention Transformer
        if self.neck is not None:
            query_feats = query_feats[0]
            support_feats = support_feats[0]
            n_repeat = support_feats.size(0) // query_feats.size(0)
            query_feats = torch.repeat_interleave(query_feats, n_repeat, dim=0)
            # print("Shapes before CAT neck:")
            # print(f"  query_feats.shape = {query_feats.shape}")
            # print(f"  support_feats.shape = {support_feats.shape}")
            query_feats, support_feats = self.neck(query_feats, support_feats)

        '''
        print("Shapes after CAT neck:")
        print(f"  query_feats.shape = {query_feats.shape}")
        print(f"  support_feats.shape = {support_feats.shape}")
        '''

        query_feats = (query_feats,)
        support_feats = (support_feats,)
        losses = dict()

        # RPN forward and loss
        # print("Computing rpn_with_support...")
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            if self.rpn_with_support:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    # query_feats,
                    # support_feats,  # TODO: delete? support_feats are not used in TransformerNeckRPNHead
                    query_feats,
                    # support_feats_cat,
                    query_img_metas=query_data['img_metas'],
                    query_gt_bboxes=query_data['gt_bboxes'],
                    query_gt_labels=None,
                    query_gt_bboxes_ignore=query_data.get(
                        'gt_bboxes_ignore', None),
                    support_img_metas=support_data['img_metas'],
                    support_gt_bboxes=support_data['gt_bboxes'],
                    support_gt_labels=support_data['gt_labels'],
                    support_gt_bboxes_ignore=support_data.get(
                        'gt_bboxes_ignore', None),
                    proposal_cfg=proposal_cfg)
            # TODO: maybe else statement need to be used.
            else:
                rpn_losses, proposal_list = self.rpn_head.forward_train(
                    query_feats,
                    copy.deepcopy(query_data['img_metas']),
                    copy.deepcopy(query_data['gt_bboxes']),
                    gt_labels=None,
                    gt_bboxes_ignore=copy.deepcopy(
                        query_data.get('gt_bboxes_ignore', None)),
                    proposal_cfg=proposal_cfg)
            # print("rpn_losses and proposal_list computed!!")
            losses.update(rpn_losses)
            # print(f"losses updated with RPN!!")
        else:
            proposal_list = proposals

        return losses
