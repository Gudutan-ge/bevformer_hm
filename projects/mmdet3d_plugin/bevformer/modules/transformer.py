# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.runner.base_module import BaseModule

from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.runner.base_module import BaseModule
from torchvision.transforms.functional import rotate
from .temporal_self_attention import TemporalSelfAttention
from .spatial_cross_attention import MSDeformableAttention3D
from .decoder import CustomMSDeformableAttention
from projects.mmdet3d_plugin.models.utils.bricks import run_time
from mmcv.runner import force_fp32, auto_fp16


@TRANSFORMER.register_module()
class PerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 **kwargs):
        super(PerceptionTransformer, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        # nn.Parameter 将一个普通的张量转换为模型的参数
        self.level_embeds = nn.Parameter(torch.Tensor( #尺度嵌入：可学习的embedding参数，num_feature_levels：特征尺度数量，embed_dims：嵌入向量的维度
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(  #摄像头嵌入：可学习的embedding参数，num_cams：摄像头数量，embed_dims：嵌入向量的维度
            self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential( # 多层感知机。处理can bus信息，can_bus 输入包含 18 个特征
            nn.Linear(18, self.embed_dims // 2), #18个输入维度
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims), # 最后映射到embed_dims维度，便于与其他bev特征进行融合
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'prev_bev', 'bev_pos'))
    def get_bev_features(
            self,
            mlvl_feats, #多尺度信息 list(1, 6, 256, 15, 25)
            bev_queries,#([50*50, 256])
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512], #每个 BEV 网格的大小，默认为 `[0.512, 0.512]`。
            bev_pos=None,
            prev_bev=None,
            **kwargs): # kwargs用于接收 任意数量的关键字参数，这里包含 img_meta——图片信息
        """
        obtain bev features.
        """

        bs = mlvl_feats[0].size(0) # bs = 1
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1) #在第 1 维扩展查询张量变为 (num_queries, 1, embed_dim)，并将其在第 1 维上重复bs (num_queries, bs, embed_dim)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1) #bev_pos从 [1, 256, 50, 50] 先变成 [1, 256, 50 * 50] 再变成 [50 * 50, 1, 256]， bevquery维度一致

        # obtain rotation angle and shift with ego motion
        # 为了得到shift
        delta_x = np.array([each['can_bus'][0]
                           for each in kwargs['img_metas']]) # kwargs['img_metas']包含每一帧的图像信息，图像信息通过字典存储，而其中的can_bus为车辆运动信息，第0个是车辆在x方向上的平移量
        delta_y = np.array([each['can_bus'][1]
                           for each in kwargs['img_metas']]) # 第1个是车辆在y方向上的平移量
        ego_angle = np.array(
            [each['can_bus'][-2] / np.pi * 180 for each in kwargs['img_metas']]) # 倒数第2个是车辆的现在的朝向角度（以弧度为单位），转化为°
        
        grid_length_y = grid_length[0] # grid_length_y 和 grid_length_x 分别是 BEV 一个网格在 y 和 x 方向上的大小（单位：米）。
        grid_length_x = grid_length[1]
        
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2) #平移距离的欧氏距离。
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180 #平移方向的角度
        bev_angle = ego_angle - translation_angle # 车辆的运动方向与当前朝向之间的相对旋转 = 当前朝向 - 运动方向
        shift_y = translation_length * \
            np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * \
            np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift # use_shift为 bool 值
        shift_x = shift_x * self.use_shift #计算 BEV 特征图在 x 和 y 方向上的偏移。
        # shift_x,shift_y是数组，包含多张图像的偏移
        shift = bev_queries.new_tensor( #确保新张量的类型（如浮点数类型）和设备（如 CPU 或 GPU）与 bev_queries 一致
            [shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy  生成一个形状为 (bs, 2) 的张量

        if prev_bev is not None: # 有 prev_bev
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs['img_metas'][i]['can_bus'][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(
                        bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle,
                                          center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(
                        bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor(
            [each['can_bus'] for each in kwargs['img_metas']])  # [:, :] (1,18) can_bus内容见 docs/img_metasss.txt
        can_bus = self.can_bus_mlp(can_bus)[None, :, :] # 多层感知机处理canbus信息，None 是在 can_bus 的第一维上插入一个新的维度，相当于 unsqueeze(0)
        bev_queries = bev_queries + can_bus * self.use_can_bus # 将 can_bus 与 bev_queries 结合


        # SCA 才会用到多尺度图像信息
        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats): # lvl是索引，feat是内容(1, 6, 256, h, w)，遍历不同尺度信息
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w) # 空间大小
            feat = feat.flatten(3).permute(1, 0, 3, 2) # (6, 1, h * w, 256)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype) # 将 self.cams_embeds 的形状从 (num_cam, embed_dim) 扩展到 (num_cam, 1, 1, embed_dim) = (6, 1, 1, 256)
                # 广播机制 将(6, 1, 1, 256)广播到(6, 1, h * w, 256)再逐元素相加。每个空间位置的每个摄像头都加入了对应的摄像头嵌入向量。
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype) 
            # lvl:lvl + 1 实际上选取的是 self.level_embeds[lvl] 对应的行（即第 lvl 个尺度的嵌入向量）表示当前尺度的嵌入向量。 (1, embed_dims)
            # None, None 将 self.level_embeds 的形状从 (1, embed_dims) 扩展到 (1, 1, 1, embed_dims)
            # 广播机制 将(1, 1, 1, 256)广播到(6, 1, h * w, 256)再逐元素相加。每个空间位置的每个摄像头都会加入对应的尺度嵌入信息。
            spatial_shapes.append(spatial_shape) # 存储不同尺度下空间形状
            feat_flatten.append(feat) # 存储不同尺度特征 list[(6, 1, h * w, 256)]

        feat_flatten = torch.cat(feat_flatten, 2) # 将列表中所有张量按第二维拼接(6, 1, h1 * w1 + h2 * w2 + ... + hn * wn, 256)，拼接不同尺度下空间位置
        spatial_shapes = torch.as_tensor( # 将不同尺度空间大小变为张量 (num_levels, 2)
            spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
            # prod(1)根据第1维度将元素相乘 tensor([h1 * w1, h2 * w2, ..., hn * wn])，cumsum(0)沿着第0维度求累计和tensor([h1 * w1, h2 * w2 + h1 * w1, ..., hn * wn + ... + h2 * w2 + h1 * w1]) [:-1]取出除最后一个以外的所有值
            # new_zeros((1,))创建形状为(1,)的值为0的张量，设备与 spatial_shapes 相同。 cat将两个张量拼接tensor([0, h1 * w1, h2 * w2 + h1 * w1, ..., hn-1 * wn-1 + ... + h2 * w2 + h1 * w1])——找到在展平的空间中每个尺度的起始位置
        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)  (6, h1 * w1 + h2 * w2 + ... + hn * wn, 1, 256)

        bev_embed = self.encoder( # 进入encoder
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs
        )

        return bev_embed

    @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    def forward(self,
                mlvl_feats,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        bev_embed = self.get_bev_features(
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(
            object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs)

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out
