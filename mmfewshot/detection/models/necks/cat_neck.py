import torch.nn as nn
from ..builder import NECKS
from ..utils.transformer_custom import CrossAttentionTransformerBlock
from torch import Tensor

@NECKS.register_module()
class CATNeck(nn.Module):
    '''
    Performs Cross Attention Transformer block between query and support features.
    Based on https://arxiv.org/abs/2104.14984

    Create a symbolic link to this file in:
    miniconda3/envs/openmmlab-new/lib/python3.7/site-packages/mmdet/models/necks
    '''


    """
    Args:
        in_channels (int): Number of input features channels.
        out_channels (int): Number of output features channels.
            Default: None.  # TODO: delete?
        num_layers (int): Number of CAT layers to use (N = 4 in the paper).
        num_heads (int): Number of heads to use in each CAT layer (M = 8 in the
            paper).
        embed_size (int): Transformer sequence dimension.
        forward_expansion (int): Factor of embed_size for the first FFN after
            multi-head attention.
        pos_encoding (bool): Either add sinusoidal positional encodings or not.
        dropout_prob (float): Dropout ratio.
        init_cfg (ConfigDict | None): Initialization config dict.
            Default: None.  # TODO: delete?

    """
    def __init__(self,
                 in_channels: int,
                 num_layers: int,
                 num_heads: int,
                 embed_size: int,
                 forward_expansion: int,
                 pos_encoding: bool,
                 dropout_prob: float = 0.,):
        # print("Building CATNeck block...")
        super(CATNeck, self).__init__()
        assert in_channels is not None, \
            "CatNeck require config of 'in_channels'."
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.embed_size = embed_size
        self.forward_expansion = forward_expansion
        self.pos_encoding = pos_encoding
        self.dropout_prob = dropout_prob
        self.cat_block = CrossAttentionTransformerBlock(in_channels=self.in_channels,
                                                        num_layers=self.num_layers,
                                                        num_heads=self.num_heads,
                                                        embed_size=self.embed_size,
                                                        forward_expansion=self.forward_expansion,
                                                        pos_encoding=self.pos_encoding,
                                                        dropout_prob=self.dropout_prob)
    # TODO: x_query goes to RPN, but x_support should be used in RoI matching?
    def forward(self, query_feat: Tensor, support_feat: Tensor) -> Tensor:
        """Calculate aggregated features of query and support.

        Args:
            query_feat (Tensor): Input query features with shape (N, C, H_q, W_q).
            support_feat (Tensor): Input support features with shape
                (N, C, H_s, W_s).

        Returns:
            Tensor: When `with_fc` is True, the aggregate feature is with
                shape (N, C), otherwise, its shape is (N, C, H_q, W_q).
        """
        '''
        print("Entering forward in CATNeck:")
        print(f"  num_layers = {self.num_layers}, num_heads = {self.num_heads}")
        print(f"  query_feat.size() = {query_feat.size()}")
        print(f"  support_feat.size() = {support_feat.size()}")
        '''
        # print("Entering forward in CATNeck...")
        assert query_feat.size(1) == support_feat.size(1), \
            'mismatch channel number between query and support features.'
        x_query = query_feat
        x_support = support_feat
        # print("Calling CrossAttentionTransformerBlock...")
        x_query, x_support = self.cat_block(x_query, x_support)

        return x_query, x_support
