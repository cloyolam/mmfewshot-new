# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, List, Optional, Tuple

import torch
from mmcv.runner import force_fp32
from mmcv.utils import ConfigDict
from mmdet.core import bbox2roi, images_to_levels, multi_apply
from mmdet.models import RPNHead
from mmdet.models.builder import HEADS, build_roi_extractor
from torch import Tensor

from mmfewshot.detection.models.utils import build_aggregator


@HEADS.register_module()
class TransformerNeckRPNHead(RPNHead):
    """RPN head for `Cross Attention Transformer <https://arxiv.org/abs/2104.14984>`_.

    Args:
        num_support_ways (int): Number of sampled classes (pos + neg).
        num_support_shots (int): Number of shot for each classes.
        aggregation_layer (dict): Config of `aggregation_layer`.
        roi_extractor (dict): Config of `roi_extractor`.
    """

    def __init__(self,
                 num_support_ways: int,
                 num_support_shots: int,
                 aggregation_layer: Dict = dict(
                     type='AggregationLayer',
                     aggregator_cfgs=[
                         dict(type='DummyAggregator',)
                     ]),
                 roi_extractor: Dict = dict(
                     type='SingleRoIExtractor',
                     roi_layer=dict(
                         type='RoIAlign', output_size=14, sampling_ratio=0),
                     out_channels=1024,
                     featmap_strides=[16]),
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_support_ways = num_support_ways
        self.num_support_shots = num_support_shots
        '''
        print("Entering constructor in TransformerRPNHead...")
        print(f"  num_support_ways = {self.num_support_ways}")
        print(f"  num_support_shots = {self.num_support_shots}")
        '''
        assert roi_extractor is not None, \
            'missing config of roi_extractor.'
        assert aggregation_layer is not None, \
            'missing config of aggregation_layer.'
        self.aggregation_layer = \
            build_aggregator(copy.deepcopy(aggregation_layer))
        self.roi_extractor = \
            build_roi_extractor(copy.deepcopy(roi_extractor))

    def extract_roi_feat(self, feats: List[Tensor], rois: Tensor) -> Tensor:
        """Forward function.

        Args:
            feats (list[Tensor]): Input features with shape (N, C, H, W).
            rois (Tensor): with shape (m, 5).

         Returns:
            Tensor: RoI features with shape (N, C, H, W).
        """
        return self.roi_extractor(feats, rois)

    # TODO: support_feats not needed if CATNeck is used? Delete those args
    def forward_train(self,
                      query_feats: List[Tensor],
                      # support_feats: List[Tensor],
                      query_gt_bboxes: List[Tensor],
                      query_img_metas: List[Dict],
                      support_gt_bboxes: List[Tensor],
                      query_gt_bboxes_ignore: Optional[List[Tensor]] = None,
                      proposal_cfg: Optional[ConfigDict] = None,
                      **kwargs) -> Tuple[Dict, List[Tuple]]:
        """Forward function in training phase.

        Args:
            query_feats (list[Tensor]): List of query features, each item
                with shape (N, C, H, W)..
            support_feats (list[Tensor]): List of support features, each item
                with shape (N, C, H, W).
            query_gt_bboxes (list[Tensor]): List of ground truth bboxes of
                query image, each item with shape (num_gts, 4).
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: `img_shape`, `scale_factor`, `flip`, and may
                also contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            support_gt_bboxes (list[Tensor]): List of ground truth bboxes of
                support image, each item with shape (num_gts, 4).
            query_gt_bboxes_ignore (list[Tensor]): List of ground truth bboxes
                to be ignored of query image with shape (num_ignored_gts, 4).
                Default: None.
            proposal_cfg (:obj:`ConfigDict`): Test / postprocessing
                configuration. if None, test_cfg would be used. Default: None.

        Returns:
            tuple: loss components and proposals of each image.

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - proposal_list (list[Tensor]): Proposals of each image.
        """

        batch_size = len(query_img_metas)
        query_feat = query_feats[0]  # (N * num_support_ways * num_support_shots, C, H_q, W_q)

        '''
        print("Entering forward_train in TransformerNeckRPNHead...")
        print(f"  batch_size = {batch_size}")                            # 2
        print(f"  self.num_support_ways = {self.num_support_ways}")      # 2
        print(f"  self.num_support_shots = {self.num_support_shots}")    # 5
        print(f"  query_feat.size() = {query_feat.size()}")              # (N * num_support_ways * num_support_shots, C, H_q, W_q)
        # print(f"  support_feats[0].size() = {support_feats[0].size()}")  # (N * num_support_ways * num_support_shots, C, H_s, W_s)
        '''

        assert batch_size == query_feat.size(0) // (self.num_support_ways * self.num_support_shots), \
            'invalid shape for query_feats.'
        # TODO: delete support_rois and support_roi_feats when using CATNeck?
        # support_rois = bbox2roi([bboxes for bboxes in support_gt_bboxes])
        # support_roi_feats = self.extract_roi_feat(support_feats, support_rois)  # (N * num_support_ways * num_support_shots, C, 14, 14)
        # print(f"  support_roi_feats.size() = {support_roi_feats.size()}")

        # TODO: add a config param to decide whether to take the average or not

        ########################################################################
        # First version: averaging over positive and negative samples before RPN
        ########################################################################

        # Luego del bloque de CrossAttention, query_feat tiene como primera
        # dimensión la cantidad de ejemplos de soporte (num_support_way *
        # num_support_shots), pero debería crear dos nuevos mapas de tamaño [1,
        # C, H_q, W_q], uno para los ejemplos positivos y otro para los
        # negativos.

        # support features are placed in follow order:
        # [pos * num_support_shots,
        #  neg * num_support_shots * (num_support_ways - 1 )] * batch size

        # get the average of queries after cross-attention,
        # from positive and negative supports.
        # If num_support_ways = 2 and batch_size = 2:
        # [(1, C, H_q, W_q)_(q1_pos), (1, C, H_q, W_q)_(q1_neg),
        #  (1, C, H_q, W_q)_(q2_pos), (1, C, H_q, W_q)_(q2_neg)]
        # (it averages all cross-attention maps in each support class)


        avg_query_feats = [
            query_feat[i * self.num_support_shots:(i + 1) *
                       self.num_support_shots].mean([0],
                                                    keepdim=True)
            for i in range(
                query_feat.size(0) // self.num_support_shots)
        ]  # [[1, 1024, 16, 16], [1, 1024, 16, 16]] when n_batch=1
        # print(f"  len(avg_query_feats) = {len(avg_query_feats)}")

        # Concat all positive pair features, [(1, C, H_q, W_q)_(q1),..., (1, C, H_q, W_q)_(qN)]
        pos_pair_feats = [
            avg_query_feats[i * self.num_support_ways]
            # for i in range(query_feat.size(0) // (self.num_support_shots * self.num_support_ways))
            for i in range(batch_size)
        ]
        # print(f"  len(pos_pair_feats) = {len(pos_pair_feats)}")  # DEBERÍA SER 1, no 2
        # for ix in range(len(pos_pair_feats)):
        #     print(f"  pos_pair_feats[{ix}] = {pos_pair_feats[ix].size()}")  # DEBERÍA SER [1, 1024, 16, 16]

        # Concat all negative pair features, [(1, C, H_q, W_q)_(q1),..., (1, C, H_q, W_q)_(qN)]
        neg_pair_feats = [
            query_feat[i * self.num_support_ways + j +1].unsqueeze(0)
            # for i in range(query_feat.size(0) // self.num_support_shots)
            for i in range(batch_size)
            for j in range(self.num_support_ways - 1)
        ]
        # print(f"  len(neg_pair_feats) = {len(neg_pair_feats)}")
        # for ix in range(len(neg_pair_feats)):
        #     print(f"  neg_pair_feats[{ix}] = {neg_pair_feats[ix].size()}")



        #################################################################################
        # Second version: without averaging over positive and negative samples before RPN
        #################################################################################

        '''
        # Split into positive and negative chunks for each query
        # [(num_support_shots, C, H_q, W_q)_pos, num_support_shots, C, H_q, W_q)_neg] * N
        # TODO: use the same variable query_feat?
        query_splits = torch.split(query_feat, self.num_support_shots, dim=0)

        # Concat all positive features into one tensor
        # [(1, C, H_q, W_q)_(q1_s1)_pos,..., (1, C, H_q, W_q)_(qN_{num_support_shots})_pos]
        # TODO: consider the case when n_ways is not 2
        pos_pair_feats = torch.concat([query_splits[ix] for ix in range(0, batch_size * self.num_support_ways, 2)])
        pos_pair_feats = torch.split(pos_pair_feats, 1, dim=0)
        # for ix in range(len(pos_pair_feats)):
        #     print(f"  pos_pair_feats[{ix}] = {pos_pair_feats[ix].size()}")  # [1, 1024, 16, 16]

        # Concat all negative features into one tensor
        # [(1, C, H_q, W_q)_(q1_s1)_neg,..., (1, C, H_q, W_q)_(qN_{num_support_shots})_neg]
        # TODO: consider the case when n_ways > 2
        neg_pair_feats = torch.concat([query_splits[ix] for ix in range(1, batch_size * self.num_support_ways, 2)])
        neg_pair_feats = torch.split(neg_pair_feats, 1, dim=0)
        # for ix in range(len(neg_pair_feats)):
        #     print(f"  neg_pair_feats[{ix}] = {neg_pair_feats[ix].size()}")  #  [1, 1024, 16, 16]

        # repeat the query_img_metas and query_gt_bboxes to match
        # the order of positive and negative pairs
        # [N * num_support_shots] * num_support_ways
        # TODO: consider the case when n_ways is not 2
        repeat_query_img_metas = [x for x in repeat_query_img_metas for i in
                                  range(self.num_support_shots)] * self.num_support_ways
        repeat_query_gt_bboxes = [x for x in repeat_query_gt_bboxes for i in
                                  range(self.num_support_shots)] * self.num_support_ways
        pair_flags.extend([False for _ in range(len(neg_pair_feats))])

        '''


        #################################################################################


        # input features for losses: [pos_pair_feats, neg_pair_feats]
        # pair_flags are used to set all the gt_label from negative pairs to
        # bg classes in losses. True means positive pairs and False means
        # negative pairs

        # add positive pairs
        pair_flags = [True for _ in range(len(pos_pair_feats))]  # N * num_support_shots
        repeat_query_img_metas = copy.deepcopy(query_img_metas)  # N
        repeat_query_gt_bboxes = copy.deepcopy(query_gt_bboxes)  # N

        # repeat the query_img_metas and query_gt_bboxes to match
        # the order of positive and negative pairs
        for i in range(batch_size):
            repeat_query_img_metas.extend([query_img_metas[i]] *
                                          (self.num_support_ways - 1))  # N * n_ways
            repeat_query_gt_bboxes.extend([query_gt_bboxes[i]] *
                                          (self.num_support_ways - 1))  # N * n_ways
            # add negative pairs
            pair_flags.extend([False] * (self.num_support_ways - 1))

        '''
        print(f"  pair_flags = {pair_flags}")
        print(f"  len(repeat_query_img_metas) = {len(repeat_query_img_metas)}")
        print(f"  repeat_query_img_metas = {repeat_query_img_metas}")
        print(f"  len(repeat_query_gt_bboxes) = {len(repeat_query_gt_bboxes)}")
        print(f"  repeat_query_gt_bboxes = {repeat_query_gt_bboxes}")
        '''

        # Call RPNHead forward method, it returns a tuple of 2 lists:
        # a tensor for anchors scores and a tensor for anchors offsets
        # ([N * num_support_shots * num_support_ways, num_anchors, H_q, W_q],
        #  [N * num_support_shots * num_support_ways, num_anchors * 4, H_q, W_q])
        outs = self([torch.cat(pos_pair_feats + neg_pair_feats)])


        # print("outs:")
        # for ix, tensor in enumerate(outs):
        #     print(f"  {ix}: {type(tensor)}, len(tensor)={len(tensor)}")


        # Concatenate RPN scores and offsets with metadata from query
        # ([N * num_support_shots * num_support_ways, num_anchors, H_q, W_q],
        #  [N * num_support_shots * num_support_ways, num_anchors * 4, H_q, W_q],
        # )
        loss_inputs = outs + (repeat_query_gt_bboxes, repeat_query_img_metas)

        '''
        print(f"type(loss_inputs) = {type(loss_inputs)}")
        print(f"len(loss_inputs) = {len(loss_inputs)}")
        print("loss_inputs:")
        for ix, lista in enumerate(loss_inputs):
            print(f"  {ix}: {type(lista)}, len(tensor)={len(lista)}")
            for ix2, tensor in enumerate(lista):
                if ix < len(loss_inputs) - 1:
                    print(f"    {ix2}: {tensor.size()}")
                else:
                    print(f"    {ix2}: {tensor}")
        '''

        losses = self.loss(
            *loss_inputs,
            gt_bboxes_ignore=query_gt_bboxes_ignore,
            pair_flags=pair_flags)
        # print(f"  Losses ready!")
        proposal_list = self.get_bboxes(
            *outs, img_metas=repeat_query_img_metas, cfg=proposal_cfg)
        return losses, proposal_list

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores: List[Tensor],
             bbox_preds: List[Tensor],
             gt_bboxes: List[Tensor],
             img_metas: List[Dict],
             gt_labels: Optional[List[Tensor]] = None,
             gt_bboxes_ignore: Optional[List[Tensor]] = None,
             pair_flags: Optional[List[bool]] = None) -> Dict:
        """Compute losses of rpn head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_anchors * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_anchors * 4, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            img_metas (list[dict]): list of image info dict where each dict
                has: `img_shape`, `scale_factor`, `flip`, and may also contain
                `filename`, `ori_shape`, `pad_shape`, and `img_norm_cfg`.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_labels (list[Tensor]): Class indices corresponding to each box.
                Default: None.
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss. Default: None
            pair_flags (list[bool]): Indicate predicted result is from positive
                pair or negative pair with shape (N). Default: None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        '''
        print("  Entering loss in TransformerNeckRPNHead...")
        print(f"    cls_scores[0].size() = {cls_scores[0].size()}")
        print(f"    bbox_preds[0].size() = {bbox_preds[0].size()}")
        print(f"    gt_bboxes[0].size() = {gt_bboxes[0].size()}")
        print(f"    img_metas = {img_metas}")
        print(f"    gt_labels = {gt_labels}")
        print(f"    gt_bboxes_ignore = {gt_bboxes_ignore}")
        print(f"    pair_flags = {pair_flags}")
        '''

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        # get anchors and training targets
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        # get the indices of negative pairs
        neg_idxes = [not f for f in pair_flags]
        num_pos_from_neg_pairs = 0
        # all the gt_labels in negative pairs will be set to background
        for lvl in range(len(labels_list)):
            num_pos_from_neg_pairs += (
                labels_list[lvl][neg_idxes] == 0).sum().item()
            labels_list[lvl][neg_idxes] = 1
            bbox_weights_list[lvl][neg_idxes] = 0
        if self.sampling:
            num_total_samples = num_total_pos + num_total_neg
        else:
            num_total_samples = num_total_pos - num_pos_from_neg_pairs
        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single Tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)
        losses_cls, losses_bbox = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples)
        return dict(loss_rpn_cls=losses_cls, loss_rpn_bbox=losses_bbox)

    # TODO: delete support_feat, not needed beacause query_feats already has
    # support info.
    def simple_test(self,
                    query_feats: List[Tensor],
                    # support_feat: Tensor,
                    query_img_metas: List[Dict],
                    rescale: bool = False) -> List[Tensor]:
        """Test function without test time augmentation.

        Args:
            query_feats (list[Tensor]): List of query features, each item with
                shape(N, C, H, W).
            support_feat (Tensor): Support features with shape (N, C, H, W).
            query_img_metas (list[dict]): List of query image info dict where
                each dict has: `img_shape`, `scale_factor`, `flip`, and may
                also contain `filename`, `ori_shape`, `pad_shape`, and
                `img_norm_cfg`. For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            rescale (bool): Whether to rescale the results.
                Default: False.

        Returns:
            List[Tensor]: Proposals of each image, each item has shape (n, 5),
                where 5 represent (tl_x, tl_y, br_x, br_y, score).
        """
        # mmdet/models/dense_heads/dense_test_mixins.py
        proposal_list = self.simple_test_rpn(query_feats, query_img_metas)
        if rescale:
            for proposals, meta in zip(proposal_list, query_img_metas):
                proposals[:, :4] /= proposals.new_tensor(meta['scale_factor'])

        return proposal_list
