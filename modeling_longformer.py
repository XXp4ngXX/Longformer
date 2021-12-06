import math

import numpy as np
import paddle
import paddlenlp
from paddle.nn import Linear, Dropout, LayerNorm, LayerList, Layer
import paddle.nn.functional as F
import paddle.nn as nn

from paddlenlp.transformers.attention_utils import _convert_param_attr_to_list, Attention, MultiHeadAttention, \
    AttentionRegistry
from paddlenlp.transformers import PretrainedModel, register_base_model

"""
由于 Longformer中的attention机制是滑动+全局，与LongFormer相比只少了随机attention，LongFormer在paddlenlp.transformers 已经实现
所以 主要参考了  https://github.com/PaddlePaddle/PaddleNLP/blob/develop/paddlenlp/transformers/LongFormer/modeling.py
以及    https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py

"""
__all__ = [

]

def mish(x):
    return x * F.tanh(F.softplus(x))


def linear_act(x):
    return x


def swish(x):
    return x * F.sigmoid(x)


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + paddle.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * paddle.pow(x, 3.0))))


ACT2FN = {
    "relu": F.relu,
    "gelu": F.gelu,
    "gelu_new": gelu_new,
    "tanh": F.tanh,
    "sigmoid": F.sigmoid,
    "mish": mish,
    "linear": linear_act,
    "swish": swish,
}


@AttentionRegistry.register("longformer")
class LongFormerSparseAttention(Attention):
    def __init__(self,
                 num_heads=1,
                 block_size=1,
                 window_size=512,
                 num_global_blocks=1):
        super(LongFormerSparseAttention,
              self).__init__(num_heads, block_size, window_size,
                             num_global_blocks)
        for k, v in locals().items():
            if k != "self":
                setattr(self, k, v)
        self.num_global_blocks_back = num_global_blocks // 2
        self.num_global_blocks_front = num_global_blocks // 2   \
                if num_global_blocks % 2 == 0                  \
                else num_global_blocks // 2 + 1

    def _get_band_mask(self, blocked_query_mask, blocked_key_mask, batch_size,
                       sequence_length):
        '''
        Return second mask: [B, 1, L-G, bs, G+W]
        '''
        GB = self.num_global_blocks_back
        GF = self.num_global_blocks_front
        G = self.num_global_blocks
        R = self.num_rand_blocks
        W = self.window_size
        bs = self.block_size
        T = sequence_length
        L = T // bs  # blocked length
        B = batch_size
        H = self.num_heads
        # G+W+R
        # query_mask: [B, L, bs]
        # key_mask: [B, L, bs]
        # [B, L-G, bs, 1] * [B, L-G, 1, G*bs] -> [B, L-G, bs, G*bs]
        temp_query_mask = paddle.reshape(blocked_query_mask[:, GF:-GB],
                                         [B, L - G, bs, 1])
        temp_key_mask_front = paddle.reshape(blocked_key_mask[:, :GF],
                                             [B, 1, 1, GF * bs])
        global_block_mask_front = paddlenlp.ops.einsum(
            "blqd,bmdk->blqk", temp_query_mask, temp_key_mask_front)

        temp_key_mask_back = paddle.reshape(blocked_key_mask[:, -GB:],
                                            [B, 1, 1, GB * bs])
        global_block_mask_back = paddlenlp.ops.einsum(
            "blqd,bmdk->blqk", temp_query_mask, temp_key_mask_back)
        # create window block mask
        key_mask_list = []
        for query_block_id in range(GF, GF + W // 2):
            left_block_id = query_block_id - W // 2
            right_block_id = query_block_id + W // 2
            zero_key_mask = paddle.zeros_like(blocked_key_mask[:, -(W - (
                right_block_id + 1 - G)):-GB])
            temp_key_mask = paddle.concat(
                [blocked_key_mask[:, GF:(right_block_id + 1)], zero_key_mask],
                axis=1)
            temp_key_mask = paddle.unsqueeze(temp_key_mask, 1)
            key_mask_list.append(temp_key_mask)
        roll_key_mask1 = paddle.concat(key_mask_list, axis=1)
        roll_key_mask1 = paddle.reshape(roll_key_mask1, [0, 0, W * bs])
        key_mask_list = []

        band_length = L - G - W // 2 * 2
        for query_block_id in range(GF + W // 2, GF + W // 2 + W):
            left_block_id = query_block_id - W // 2
            right_block_id = query_block_id + W // 2
            key_mask_list.append(blocked_key_mask[:, left_block_id:left_block_id
                                                  + band_length])
        window_key_mask = paddle.concat(key_mask_list, axis=2)
        window_key_mask = paddle.reshape(window_key_mask, [0, 0, W * bs])

        key_mask_list = []
        for query_block_id in range((L - GB) - W // 2, L - GB):
            left_block_id = query_block_id - W // 2
            right_block_id = query_block_id + W // 2
            zero_key_mask = paddle.zeros_like(blocked_key_mask[:, GF:GF + W - (
                L - left_block_id - GB)])
            temp_key_mask = paddle.concat(
                [zero_key_mask, blocked_key_mask[:, left_block_id:-GB]], axis=1)
            temp_key_mask = paddle.unsqueeze(temp_key_mask, 1)
            key_mask_list.append(temp_key_mask)
        roll_key_mask2 = paddle.concat(key_mask_list, axis=1)
        roll_key_mask2 = paddle.reshape(roll_key_mask2, [0, 0, W * bs])

        window_key_mask = paddle.concat(
            [roll_key_mask1, window_key_mask, roll_key_mask2], axis=1)
        window_key_mask = paddle.unsqueeze(window_key_mask, axis=2)
        # [B, L-G, bs, 1] * [B, L-G, 1, W*bs] -> [B, L-G, bs, W*bs]
        window_block_mask = paddlenlp.ops.einsum(
            "blkd,bldq->blkq", temp_query_mask, window_key_mask)
        band_mask = paddle.concat(
            [
                global_block_mask_front, window_block_mask,
                global_block_mask_back
            ],
            axis=3)
        band_mask = paddle.unsqueeze(band_mask, 1)  # for head
        band_mask = paddle.expand(band_mask, [B, H, L - G, bs, -1])
        return band_mask

    def _get_band_matrix(self, blocked_matrix, B, T):
        '''
        return global and window matrix: [B, H, L-G, (G+W) * bs, -1]
        '''
        # blocked_matrix: [B, H, L, bs, -1]
        GB = self.num_global_blocks_back
        GF = self.num_global_blocks_front
        G = self.num_global_blocks
        R = self.num_rand_blocks
        W = self.window_size
        bs = self.block_size
        L = T // bs  # blocked length
        H = self.num_heads

        # get roll matrix
        blocked_list = []
        for query_block_id in range(GF, GF + W // 2):
            left_block_id = query_block_id - W // 2
            right_block_id = query_block_id + W // 2
            temp_blocked_matrix_list = [
                blocked_matrix[:, :, 0:(right_block_id + 1)],
                blocked_matrix[:, :, -(G + W - right_block_id - 1):]
            ]
            temp_blocked_matrix = paddle.concat(
                temp_blocked_matrix_list, axis=2)
            temp_blocked_matrix = paddle.unsqueeze(temp_blocked_matrix, axis=2)
            blocked_list.append(temp_blocked_matrix)

        # get window matrix
        band_length = L - G - W // 2 * 2
        band_matrix_list = []
        for query_block_id in range(GF + W // 2, GF + W // 2 + W):
            left_block_id = query_block_id - W // 2
            right_block_id = query_block_id + W // 2
            band_matrix_list.append(
                paddle.unsqueeze(
                    blocked_matrix[:, :, left_block_id:left_block_id +
                                   band_length],
                    axis=3))
        band_matrix = paddle.concat(band_matrix_list, axis=3)

        global_blocked_front_matrix = paddle.unsqueeze(
            blocked_matrix[:, :, :GF], axis=2)
        global_blocked_front_matrix = paddle.expand(
            global_blocked_front_matrix, [B, H, band_length, GF, bs, -1])
        global_blocked_back_matrix = paddle.unsqueeze(
            blocked_matrix[:, :, -GB:], axis=2)
        global_blocked_back_matrix = paddle.expand(
            global_blocked_back_matrix, [B, H, band_length, GB, bs, -1])
        band_matrix = paddle.concat(
            [
                global_blocked_front_matrix, band_matrix,
                global_blocked_back_matrix
            ],
            axis=3)
        blocked_list.append(band_matrix)

        for query_block_id in range(L - GB - W // 2, L - GB):
            left_block_id = query_block_id - W // 2
            right_block_id = query_block_id + W // 2
            temp_blocked_matrix_list = [
                blocked_matrix[:, :, 0:G + W - (L - left_block_id)],
                blocked_matrix[:, :, left_block_id:]
            ]
            temp_blocked_matrix = paddle.concat(
                temp_blocked_matrix_list, axis=2)
            temp_blocked_matrix = paddle.unsqueeze(temp_blocked_matrix, axis=2)
            blocked_list.append(temp_blocked_matrix)

        band_matrix = paddle.concat(blocked_list, axis=2)
        band_matrix = paddle.reshape(band_matrix,
                                     [B, H, L - G, (G + W) * bs, -1])
        return band_matrix

    # LongFormer不需要随机的attention块
    # def _get_rand_mask(self, blocked_query_mask, blocked_key_mask,
    #                    rand_mask_idx, batch_size, sequence_length):
    #     '''
    #     return random mask: [B, H, L-G, bs, R * bs]
    #     '''
    #     # rand_mask_idx: [H, T]
    #     # blocked_query_mask: [B, L, bs]
    #     # blocked_key_mask: [B, L, bs]
    #     bs = self.block_size
    #     B = batch_size
    #     L = sequence_length // bs
    #     H = self.num_heads
    #     G = self.num_global_blocks
    #     GB = self.num_global_blocks_back
    #     GF = self.num_global_blocks_front
    #     R = self.num_rand_blocks
    #     temp_block_key_mask = paddle.unsqueeze(blocked_key_mask, 1)
    #     temp_block_key_mask = paddle.expand(temp_block_key_mask, [B, H, L, -1])
    #     temp_block_key_mask_list = [
    #         paddle.gather_nd(temp_block_key_mask[b], rand_mask_idx)
    #         for b in range(B)
    #     ]
    #     temp_block_key_mask = paddle.concat(temp_block_key_mask_list, 0)
    #     temp_block_key_mask = paddle.reshape(temp_block_key_mask, [
    #         B, temp_block_key_mask.shape[0] // B // (L - GF - GB) // R,
    #         L - GF - GB, -1
    #     ])
    #     rand_mask = paddlenlp.ops.einsum("blq,bhlk->bhlqk",
    #                                      blocked_query_mask[:, GF:-GB],
    #                                      temp_block_key_mask)
    #     return rand_mask

    # def _gather_random_key_value(self, blocked_matrix, rand_mask_idx, B, T):
    #     '''
    #     return random key matrix: [B, H, L-G, R * bs, -1]
    #     '''
    #     # blocked_matrix: [B, H, L, bs, -1]
    #     # rand_mask_idx: [H, T]
    #     G = self.num_global_blocks
    #     H = self.num_heads
    #     bs = self.block_size
    #     L = T // bs
    #     R = self.num_rand_blocks
    #     gathered_matrix = paddle.concat(
    #         [
    #             paddle.gather_nd(blocked_matrix[b, :], rand_mask_idx)
    #             for b in range(B)
    #         ],
    #         axis=0)
    #     gathered_matrix = paddle.reshape(gathered_matrix,
    #                                      [B, H, L - G, R * bs, -1])
    #     return gathered_matrix

    def _get_global_out(self,
                        query_matrix,
                        key_matrix,
                        value_matrix,
                        key_mask,
                        d_head,
                        dropout,
                        is_front=True):
        GB = self.num_global_blocks_back
        GF = self.num_global_blocks_front
        if is_front:
            global_query_matrix = query_matrix[:, :, 0:GF * self.block_size]
        else:
            global_query_matrix = query_matrix[:, :, -GB * self.block_size:]
        global_product = paddle.matmul(
            global_query_matrix, key_matrix, transpose_y=True)
        global_product = global_product * (d_head**-0.5)
        global_product += (1 - key_mask) * -1e6
        global_weights = F.softmax(global_product)
        # [B, H, GF*bs, T] * [B, H, T, D] -> [B, H, GF*bs, D]
        global_product = paddle.matmul(global_weights, value_matrix)
        return global_product

    def _get_splited_matrix(self, matrix):
        W = self.window_size // 2
        return matrix[:, :, 0:W], matrix[:, :, W:-W], matrix[:, :, -W:]

    def forward(self,
                query_matrix,
                key_matrix,
                value_matrix,
                d_head,
                attn_mask=None,
                query_mask=None,
                key_mask=None,
                dropout=None):
        '''
            query_matrix: [B, H, T, D]
            key_matrix: [B, H, T, D]
            value_matrix: [B, H, T, D]
            query_mask: [B, 1, T, 1]  bool mask
            key_mask: [B, 1, 1, T]    bool mask
            Global Attention
            Window Attention            
        '''
        B = query_matrix.shape[0]  # batch_size
        H = self.num_heads
        T = query_matrix.shape[2]  # sequence_length
        D = query_matrix.shape[3]  # size per head
        G = self.num_global_blocks
        GB = self.num_global_blocks_back
        GF = self.num_global_blocks_front
        R = self.num_rand_blocks
        W = self.window_size
        bs = self.block_size
        L = T // bs  # blocked length

        blocked_query_matrix = paddle.reshape(query_matrix, [B, H, L, bs, -1])
        blocked_key_matrix = paddle.reshape(key_matrix, [B, H, L, bs, -1])
        blocked_value_matrix = paddle.reshape(value_matrix, [B, H, L, bs, -1])
        blocked_query_mask = paddle.reshape(query_mask, [B, L, bs])
        blocked_key_mask = paddle.reshape(key_mask, [B, L, bs])

        # 1. global_front_product 
        global_front_out = self._get_global_out(
            query_matrix, key_matrix, value_matrix, key_mask, d_head, dropout)

        # 2. global_back_product
        global_back_out = self._get_global_out(query_matrix, key_matrix,
                                               value_matrix, key_mask, d_head,
                                               dropout, False)

        # 3. second_product

        # create second matrix
        # [B, 1, L-G, bs, (G+W)*bs]
        band_mask = self._get_band_mask(blocked_query_mask, blocked_key_mask, B,
                                        T)
        # # [B, H, L-G, bs, R*bs]
        # rand_mask = self._get_rand_mask(blocked_query_mask, blocked_key_mask,
        #                                 rand_mask_idx, B, T)
        # [B, H, L-G, bs, (G+W+R)*bs]
        # second_mask = paddle.concat([band_mask, rand_mask], axis=4)

        # [B, H, L-G, R * bs, -1]
        # random_keys = self._gather_random_key_value(blocked_key_matrix,
        #                                            rand_mask_idx, B, T)
        # random_values = self._gather_random_key_value(blocked_value_matrix,
        #                                              rand_mask_idx, B, T)

        band_keys_matrix = self._get_band_matrix(blocked_key_matrix, B, T)
        band_value_matrix = self._get_band_matrix(blocked_value_matrix, B, T)

        # [B, H, L - G, bs, -1]
        second_query_matrix = blocked_query_matrix[:, :, GF:-GB]
        # [B, H, L - G, (G+W+R)*bs, -1]
        # second_key_matrix = paddle.concat(
        #    [band_keys_matrix, random_keys], axis=3)
        # [B, H, L - G, (G+W+R)*bs, -1]
        # second_value_matrix = paddle.concat(
        #    [band_value_matrix, random_values], axis=3)
        second_top_value_matrix, second_middle_value_matrix, second_bottom_value_matrix = \
            self._get_splited_matrix(band_value_matrix)
        second_product = paddlenlp.ops.einsum(
            "bhlqd,bhlkd->bhlqk", second_query_matrix, band_keys_matrix)
        second_product = second_product * (d_head**-0.5)
        second_product += (1 - band_mask) * -1e6
        second_weights = F.softmax(second_product)

        second_top_weights, second_middle_weights, second_bottom_weights = \
            self._get_splited_matrix(second_weights)
        second_top_out = paddlenlp.ops.einsum(
            "bhlqk,bhlkd->bhlqd", second_top_weights, second_top_value_matrix)

        second_middle_out = paddlenlp.ops.einsum(
            "bhlqk,bhlkd->bhlqd",
            second_middle_weights[:, :, :, :, GF * bs:-(GB + R) * bs],
            second_middle_value_matrix[:, :, :, GF * bs:-(GB + R) * bs])
        # add global block attention
        second_middle_out += paddlenlp.ops.einsum(
            "bhlqk,bhkd->bhlqd", second_middle_weights[:, :, :, :, :GF * bs],
            blocked_value_matrix[:, :, 0])
        second_middle_out += paddlenlp.ops.einsum(
            "bhlqk,bhkd->bhlqd",
            second_middle_weights[:, :, :, :, -(GB + R) * bs:-R * bs],
            blocked_value_matrix[:, :, -GB])
        # add random block attention
        # second_middle_out += paddlenlp.ops.einsum(
        #    "...qk,...kd->...qd", second_middle_weights[:, :, :, :, -R * bs:],
        #    random_values[:, :, GF:-GB])

        second_bottom_out = paddlenlp.ops.einsum("bhlqk,bhlkd->bhlqd",
                                                 second_bottom_weights,
                                                 second_bottom_value_matrix)

        second_out = paddle.concat(
            [second_top_out, second_middle_out, second_bottom_out], axis=2)
        second_out = paddle.reshape(second_out, [B, H, (L - G) * bs, -1])

        # [B, H, T, D]
        out = paddle.concat(
            [global_front_out, second_out, global_back_out], axis=2)
        out = out * query_mask
        return out



class TransformerEncoderLayer(Layer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward,
                 dropout=0.1,
                 activation="relu",
                 attn_dropout=None,
                 act_dropout=None,
                 normalize_before=False,
                 weight_attr=None,
                 bias_attr=None,
                 attention_type="longformer",
                 block_size=1,
                 window_size=3,
                 num_global_blocks=1,
                 num_rand_blocks=1,
                 seed=None):
        self._config = locals()
        self._config.pop("self")
        self._config.pop("__class__", None)  # py3

        super(TransformerEncoderLayer, self).__init__()
        attn_dropout = dropout if attn_dropout is None else attn_dropout
        act_dropout = dropout if act_dropout is None else act_dropout
        self.normalize_before = normalize_before

        weight_attrs = _convert_param_attr_to_list(weight_attr, 2)
        bias_attrs = _convert_param_attr_to_list(bias_attr, 2)

        self.self_attn = MultiHeadAttention(
            d_model,
            nhead,
            dropout=attn_dropout,
            weight_attr=weight_attrs[0],
            bias_attr=bias_attrs[0],
            attention_type=attention_type,
            block_size=block_size,
            window_size=window_size,
            num_global_blocks=num_global_blocks,
            num_rand_blocks=num_rand_blocks,
            seed=seed)
        self.linear1 = Linear(
            d_model, dim_feedforward, weight_attrs[1], bias_attr=bias_attrs[1])
        self.dropout = Dropout(act_dropout, mode="upscale_in_train")
        self.linear2 = Linear(
            dim_feedforward, d_model, weight_attrs[1], bias_attr=bias_attrs[1])
        self.norm1 = LayerNorm(d_model, epsilon=1e-12)
        self.norm2 = LayerNorm(d_model, epsilon=1e-12)
        self.dropout1 = Dropout(dropout, mode="upscale_in_train")
        self.dropout2 = Dropout(dropout, mode="upscale_in_train")
        self.activation = getattr(F, activation)
        self.d_model = d_model

    def forward(self,
                src,
                src_mask=None,
                rand_mask_idx=None,
                query_mask=None,
                key_mask=None):
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        src = self.self_attn(src, src, src, src_mask, rand_mask_idx, query_mask,
                             key_mask)
        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)
        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(Layer):
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = LayerList([(encoder_layer if i == 0 else
                                  type(encoder_layer)(**encoder_layer._config))
                                 for i in range(num_layers)])
        self.num_layers = num_layers
        self.norm = LayerNorm(self.layers[0].d_model, epsilon=1e-12)
        self.normalize_before = self.layers[0].normalize_before

    def forward(self,
                src,
                src_mask_list=None,
                rand_mask_idx_list=None,
                query_mask=None,
                key_mask=None):
        output = src
        if not self.normalize_before:
            output = self.norm(output)

        for i, mod in enumerate(self.layers):
            rand_mask_id = None
            if rand_mask_idx_list is not None:
                rand_mask_id = rand_mask_idx_list[i]
            if src_mask_list is None:
                output = mod(output, None, rand_mask_id, query_mask, key_mask)
            else:
                output = mod(output, src_mask_list[i], rand_mask_id, query_mask,
                             key_mask)

        if self.normalize_before:
            output = self.norm(output)
        return output


class LongFormerPooler(Layer):
    """
    Pool the result of LongFormer Encoder
    """

    def __init__(self, hidden_size):
        super(LongFormerPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LongFormerEmbeddings(Layer):
    """
    Include embeddings from word, position and token_type embeddings
    """

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 padding_idx=0):
        super(LongFormerEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=padding_idx)
        self.position_embeddings = nn.Embedding(max_position_embeddings,
                                                hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class LongFormerPretrainedModel(PretrainedModel):
    """
    An abstract class for pretrained LongFormer models. It provides LongFormer related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    model_config_file = "model_config.json"
    pretrained_init_configuration = {
        
    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        
    }
    base_model_prefix = "LongFormer"

    def init_weights(self, layer):
        # Initialization hook
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # In the dygraph mode, use the `set_value` to reset the parameter directly,
            # and reset the `state_dict` to update parameter in static mode.
            if isinstance(layer.weight, paddle.Tensor):
                layer.weight.set_value(
                    paddle.tensor.normal(
                        mean=0.0,
                        std=self.initializer_range
                        if hasattr(self, "initializer_range") else
                        self.LongFormer.config["initializer_range"],
                        shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class LongFormerModel(LongFormerPretrainedModel):
    """
    The bare LongFormer Model outputting raw hidden-states.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.
    Args:
        num_layers (int):
            Number of hidden layers in the Transformer encoder.
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `LongFormerModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `LongFormerModel`.
        nhead (int):
            Number of attention heads for each attention layer in the Transformer encoder.
        attn_dropout (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        dim_feedforward (int, optional):
            Dimensionality of the feed-forward (ff) layer in the Transformer encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `3072`.
        activation (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"``, ``"silu"`` and ``"gelu_new"`` are supported.
            Defaults to `"gelu"`.
        normalize_before (bool, optional):
            Indicates whether to put layer normalization into preprocessing of MHA and FFN sub-layers.
            If True, pre-process is layer normalization and post-precess includes dropout,
            residual connection. Otherwise, no pre-process and post-precess includes dropout,
            residual connection, layer normalization.
            Defaults to `False`.
        block_size (int, optional):
            The block size for the attention mask.
            Defaults to `1`.
        window_size (int, optional):
            The number of block in a window.
            Defaults to `3`.
        num_global_blocks (int, optional):
            Number of global blocks per sequence.
            Defaults to `1`.
        num_rand_blocks (int, optional):
            Number of random blocks per row.
            Defaults to `2`.
        seed (int, optional):
            The random seed for generating random block id.
            Defaults to ``None``.
        pad_token_id (int, optional):
            The index of padding token for LongFormer embedding.
            Defaults to ``0``.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layer and pooler layer.
            Defaults to `768`.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids`.
            Defaults to `2`.
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 nhead,
                 attn_dropout=0.1,
                 dim_feedforward=3072,
                 activation="gelu",
                 normalize_before=False,
                 block_size=1,
                 window_size=3,
                 num_global_blocks=1,
                 num_rand_blocks=2,
                 seed=None,
                 pad_token_id=0,
                 hidden_size=768,
                 hidden_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 **kwargs):
        super(LongFormerModel, self).__init__()
        # embedding
        self.embeddings = LongFormerEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size, pad_token_id)

        # encoder
        encoder_layer = TransformerEncoderLayer(
            hidden_size,
            nhead,
            dim_feedforward,
            attn_dropout,
            activation,
            normalize_before=normalize_before,
            attention_type="LongFormer",
            block_size=block_size,
            window_size=window_size,
            num_global_blocks=num_global_blocks,
            num_rand_blocks=num_rand_blocks,
            seed=seed)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        # pooler
        self.pooler = LongFormerPooler(hidden_size)
        self.pad_token_id = pad_token_id
        self.num_layers = num_layers

    def _process_mask(self, input_ids, attention_mask_list=None):
        # [B, T]
        attention_mask = (input_ids == self.pad_token_id
                          ).astype(self.pooler.dense.weight.dtype)
        # [B, 1, T, 1]
        query_mask = paddle.unsqueeze(attention_mask, axis=[1, 3])
        # [B, 1, 1, T]
        key_mask = paddle.unsqueeze(attention_mask, axis=[1, 2])
        query_mask = 1 - query_mask
        key_mask = 1 - key_mask
        return attention_mask_list, query_mask, key_mask

    def forward(self,
                input_ids,
                token_type_ids=None,
                attention_mask_list=None,
                rand_mask_idx_list=None):
        r"""
        The LongFormerModel forward method, overrides the __call__() special method.
        Args:
            input_ids (`Tensor`):
                Indices of input sequence tokens in the vocabulary.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
            token_type_ids (`Tensor`, optional):
                Segment token indices to indicate first and second portions of the inputs.
                Indices can either be 0 or 1:
                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.
                Its data type should be `int64` and it has a shape of [batch_size, sequence_length].
                Defaults to ``None``, which means we don't add segment embeddings.
            attention_mask_list (list, optional):
                A list which contains some tensors used in multi-head attention
                to prevents attention to some unwanted positions, usually the
                paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, n_head, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Defaults to `None`, which means nothing needed to be prevented attention to.
            rand_mask_idx_list (`list`, optional):
                A list which contains some tensors used in LongFormer random block.
        Returns:
            tuple: Returns tuple (`encoder_output`, `pooled_output`).
            With the fields:
            - encoder_output (Tensor):
                Sequence of output at the last layer of the model.
                Its data type should be float32 and has a shape of [batch_size, sequence_length, hidden_size].
            - pooled_output (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].
        Examples:
            .. code-block::
                import paddle
                from paddlenlp.transformers import LongFormerModel, LongFormerTokenizer
                from paddlenlp.transformers import create_LongFormer_rand_mask_idx_list
                tokenizer = LongFormerTokenizer.from_pretrained('LongFormer-base-uncased')
                model = LongFormerModel.from_pretrained('LongFormer-base-uncased')
                config = model.config
                max_seq_len = 512
                input_ids = tokenizer.convert_tokens_to_ids(
                    tokenizer(
                        "This is a docudrama story on the Lindy Chamberlain case and a look at "
                        "its impact on Australian society It especially looks at the problem of "
                        "innuendo gossip and expectation when dealing with reallife dramasbr br "
                        "One issue the story deals with is the way it is expected people will all "
                        "give the same emotional response to similar situations Not everyone goes "
                        "into wild melodramatic hysterics to every major crisis Just because the "
                        "characters in the movies and on TV act in a certain way is no reason to "
                        "expect real people to do so"
                    ))
                input_ids.extend([0] * (max_seq_len - len(input_ids)))
                seq_len = len(input_ids)
                input_ids = paddle.to_tensor([input_ids])
                rand_mask_idx_list = create_LongFormer_rand_mask_idx_list(
                    config["num_layers"], seq_len, seq_len, config["nhead"],
                    config["block_size"], config["window_size"], config["num_global_blocks"],
                    config["num_rand_blocks"], config["seed"])
                rand_mask_idx_list = [
                    paddle.to_tensor(rand_mask_idx) for rand_mask_idx in rand_mask_idx_list
                ]
                output = model(input_ids, rand_mask_idx_list=rand_mask_idx_list)
        """
        embedding_output = self.embeddings(input_ids, token_type_ids)
        attention_mask_list, query_mask, key_mask = self._process_mask(
            input_ids, attention_mask_list)
        encoder_output = self.encoder(embedding_output, attention_mask_list,
                                      rand_mask_idx_list, query_mask, key_mask)
        pooled_output = self.pooler(encoder_output)
        return encoder_output, pooled_output


''' this is early version based on huggingface not use paddlenlp.transformers.attention_utils
class LongformerEmbeddings(Layer):
    def __init__(
        self,
        vocab_size,
        hidden_size=768, hidden_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        pad_token_id=0
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position_embeddings = nn.Embedding(
            max_position_embeddings, hidden_size, padding_idx=pad_token_id)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        if position_ids is None:
            # maybe need use shape op to unify static graph and dynamic graph
            ones = paddle.ones_like(input_ids, dtype="int64")
            seq_length = paddle.cumsum(ones, axis=-1)
            position_ids = seq_length - ones
            position_ids.stop_gradient = True
        if token_type_ids is None:
            token_type_ids = paddle.zeros_like(input_ids, dtype="int64")

        input_embedings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_embedings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class LongformerSelfAttention(Layer):

    """
    Longformer SelfAttention Layer With Sliding Window Attention And Global Attention
    """

    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=3,
        attention_probs_dropout_probs=0.1,
        attention_window: Union[List[int], int] = 512,
        layer_id=None
    ):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({hidden_size}) is not a multiple of the number of attention "
                f"heads ({num_attention_heads})"
            )

        self.num_heads = num_attention_heads
        self.head_dim = int(hidden_size / num_attention_heads)
        self.embed_dim = hidden_size

        self.query = nn.Linear(hidden_size, self.embed_dim)
        self.key = nn.Linear(hidden_size, self.embed_dim)
        self.value = nn.Linear(hidden_size, self.embed_dim)

        # separate projection layers for tokens with global attention
        self.query_global = nn.Linear(hidden_size, self.embed_dim)
        self.key_global = nn.Linear(hidden_size, self.embed_dim)
        self.value_global = nn.Linear(hidden_size, self.embed_dim)

        self.dropout = attention_probs_dropout_probs

        self.layer_id = layer_id
        attention_window = attention_window[self.layer_id]
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        """
        :class:`LongformerSelfAttention` expects `len(hidden_states)` to be multiple of `attention_window`. Padding to
        `attention_window` happens in :meth:`LongformerModel.forward` to avoid redoing the padding on each layer.
        The `attention_mask` is changed in :meth:`LongformerModel.forward` from 0, 1, 2 to:
            * -10000: no attention
            * 0: local attention
            * +10000: global attention
        """


class LongformerSelfOutput(Layer):
    def __init__(
        self,
        hidden_size,
        hidden_dropout_prob
    ):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class LongformerAttention(Layer):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_probs,
        attention_window,
        hidden_dropout_prob,
        layer_id
    ):
        super().__init__()
        self.self = LongformerSelfAttention(hidden_size=hidden_size,
                                            num_attention_heads=num_attention_heads,
                                            attention_probs_dropout_prob=attention_probs_dropout_probs,
                                            attention_window=attention_window,
                                            layer_id=layer_id)
        self.output = LongformerSelfOutput(
            hidden_size=hidden_size, hidden_dropout_prob=hidden_dropout_prob)
        # self.pruned_heads = set() 指定层的注意力头剪枝 不实现

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output,) + self_outputs[1:]
        return outputs


class LongformerIntermediate(Layer):
    def __init__(self, hidden_size, intermediate_size, hidden_act):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size, hidden_act)
        if isinstance(hidden_act, str):
            self.intermediate_act_fn = ACT2FN[hidden_act]
        else:
            self.intermediate_act_fn = hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput
class LongformerOutput(nn.Layer):
    def __init__(self, intermediate_size, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# this is the layer of Longformer
# ref: https://github.com/huggingface/transformers/blob/master/src/transformers/models/longformer/modeling_longformer.py class LongformerLayer
class LongformerLayer(nn.Layer):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_probs,
        attention_window,
        hidden_dropout_prob,
        intermediate_size,
        hidden_act,
        layer_id=0
    ):
        super().__init__()
        self.attention = LongformerAttention(
            hidden_size,
            num_attention_heads,
            attention_probs_dropout_probs,
            attention_window,
            hidden_dropout_prob,
            layer_id
        )
        self.intermediate = LongformerIntermediate(
            hidden_size, intermediate_size, hidden_act)
        self.output = LongformerOutput(
            intermediate_size, hidden_size, hidden_dropout_prob)
        # self.chunk_size_feed_forward = config.chunk_size_feed_forward 分块 不实现
        # self.seq_len_dim = 1

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        layer_head_mask=None,
        is_index_masked=None,
        is_index_global_attn=None,
        is_global_attn=None,
        output_attentions=False,
    ):
        self_attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
            output_attentions=output_attentions,
        )
        attn_output = self_attn_outputs[0]
        outputs = self_attn_outputs[1:]

        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)

        outputs = (layer_output,) + outputs
        return outputs


class LongformerEncoder(Layer):
    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        attention_probs_dropout_probs,
        attention_window,
        hidden_dropout_prob,
        intermediate_size,
        hidden_act,
        num_hidden_layers=24
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_probs_dropout_probs = attention_probs_dropout_probs
        self.attention_window = attention_window
        self.hidden_dropout_prob = hidden_dropout_prob
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.num_hidden_layers = num_hidden_layers
        self.layer = nn.LayerList(
            [LongformerLayer(
                hidden_size,
                num_attention_heads,
                attention_probs_dropout_probs,
                attention_window,
                hidden_dropout_prob,
                intermediate_size,
                hidden_act,
                layer_id=i
            ) for i in range(self.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False
    ):

        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()

        all_hidden_states = () if output_hidden_states else None
        # All local attentions.
        all_attentions = () if output_attentions else None
        all_global_attentions = () if (output_attentions and is_global_attn) else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            assert head_mask.size()[0] == (
                len(self.layer)
            ), f"The head_mask should be specified for {len(self.layer)} layers, but it is for {head_mask.size()[0]}."
        for idx, sub_layer in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 训练时用到的gradient_checkpointing trick 不作实现
            # if self.gradient_checkpointing and self.training:

            #     def create_custom_forward(module):
            #         def custom_forward(*inputs):
            #             return module(*inputs, is_global_attn, output_attentions)

            #         return custom_forward

            #     layer_outputs = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(sub_layer),
            #         hidden_states,
            #         attention_mask,
            #         head_mask[idx] if head_mask is not None else None,
            #         is_index_masked,
            #         is_index_global_attn,
            #     )

            layer_outputs = sub_layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=head_mask[idx] if head_mask is not None else None,
                is_index_masked=is_index_masked,
                is_index_global_attn=is_index_global_attn,
                is_global_attn=is_global_attn,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]

            if output_attentions:
                # bzs x seq_len x num_attn_heads x (num_global_attn + attention_window_len + 1) => bzs x num_attn_heads x seq_len x (num_global_attn + attention_window_len + 1)
                all_attentions = all_attentions + \
                    (layer_outputs[1].transpose(1, 2),)

                if is_global_attn:
                    # bzs x num_attn_heads x num_global_attn x seq_len => bzs x num_attn_heads x seq_len x num_global_attn
                    all_global_attentions = all_global_attentions + \
                        (layer_outputs[2].transpose(2, 3),)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v for v in [hidden_states, all_hidden_states, all_attentions, all_global_attentions] if v is not None
        )


# copy from transformers.models.roberta.modeling.py with Roberta->Longformer
class LongformerPooler(nn.Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class LongformerPretrainedModel(PretrainedModel):
    r"""
    An abstract class for pretrained Longformer models. It provides Longformer related
    `model_config_file`, `pretrained_init_configuration`, `resource_files_names`,
    `pretrained_resource_files_map`, `base_model_prefix` for downloading and
    loading pretrained models.
    See :class:`~paddlenlp.transformers.model_utils.PretrainedModel` for more details.
    """

    # model config in saving and loading in local file system
    model_config_file = "model_config.json"
    # this project dont use built-in pretrained models
    pretrained_init_configuration = {

    }
    resource_files_names = {"model_state": "model_state.pdparams"}
    pretrained_resource_files_map = {
        "model_state": {
        }
    }
    base_model_prefix = "longformer"

    def init_weights(self, layer):
        """ Initialization hook """
        if isinstance(layer, (nn.Linear, nn.Embedding)):
            # only support dygraph, use truncated_normal and make it inplace
            # and configurable later
            layer.weight.set_value(
                paddle.tensor.normal(
                    mean=0.0,
                    std=self.initializer_range
                    if hasattr(self, "initializer_range") else
                    self.longformer.config["initializer_range"],
                    shape=layer.weight.shape))
        elif isinstance(layer, nn.LayerNorm):
            layer._epsilon = 1e-12


@register_base_model
class LongformerModel(LongformerPretrainedModel):
    r"""
    The bare Longformer Model outputting raw hidden-states.
    This model inherits from :class:`~paddlenlp.transformers.model_utils.PretrainedModel`.
    Refer to the superclass documentation for the generic methods.
    This model is also a Paddle `paddle.nn.Layer <https://www.paddlepaddle.org.cn/documentation
    /docs/en/api/paddle/fluid/dygraph/layers/Layer_en.html>`__ subclass. Use it as a regular Paddle Layer
    and refer to the Paddle documentation for all matter related to general usage and behavior.
    Args:
        vocab_size (int):
            Vocabulary size of `inputs_ids` in `LongformerModel`. Also is the vocab size of token embedding matrix.
            Defines the number of different tokens that can be represented by the `inputs_ids` passed when calling `LongformerModel`.
        hidden_size (int, optional):
            Dimensionality of the embedding layer, encoder layers and pooler layer. Defaults to `768`.
        num_hidden_layers (int, optional):
            Number of hidden layers in the Transformer encoder. Defaults to `12`.
        num_attention_heads (int, optional):
            Number of attention heads for each attention layer in the Transformer encoder.
            Defaults to `12`.
        intermediate_size (int, optional):
            Dimensionality of the feed-forward (ff) layer in the encoder. Input tensors
            to ff layers are firstly projected from `hidden_size` to `intermediate_size`,
            and then projected back to `hidden_size`. Typically `intermediate_size` is larger than `hidden_size`.
            Defaults to `3072`.
        hidden_act (str, optional):
            The non-linear activation function in the feed-forward layer.
            ``"gelu"``, ``"relu"`` and any other paddle supported activation functions
            are supported. Defaults to ``"gelu"``.
        hidden_dropout_prob (float, optional):
            The dropout probability for all fully connected layers in the embeddings and encoder.
            Defaults to `0.1`.
        attention_probs_dropout_prob (float, optional):
            The dropout probability used in MultiHeadAttention in all encoder layers to drop some attention target.
            Defaults to `0.1`.
        max_position_embeddings (int, optional):
            The maximum value of the dimensionality of position encoding, which dictates the maximum supported length of an input
            sequence. Defaults to `512`.
        type_vocab_size (int, optional):
            The vocabulary size of the `token_type_ids` passed when calling `~transformers.LongformerModel`.
            Defaults to `2`.
        initializer_range (float, optional):
            The standard deviation of the normal initializer. Defaults to 0.02.
            .. note::
                A normal_initializer initializes weight matrices as normal distributions.
                See :meth:`LongformerPretrainedModel._init_weights()` for how weights are initialized in `LongformerModel`.
        pad_token_id(int, optional):
            The index of padding token in the token vocabulary.
            Defaults to `0`.
    """

    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        num_hidden_layers=24,
        num_attention_heads=12,
        attention_window: Union[List[int], int] = 512,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=16,
        initializer_range=0.02,
        pad_token_id=0
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.initializer_range = initializer_range
        self.embeddings = LongformerEmbeddings(
            vocab_size, hidden_size, hidden_dropout_prob,
            max_position_embeddings, type_vocab_size, pad_token_id)
        # there is the main differnce because the Longformer encoder uses attention window and global attention
        self.encoder = LongformerEncoder(
            hidden_size=hidden_size, num_attention_heads=num_attention_heads, attention_probs_dropout_probs=attention_probs_dropout_prob,
            attention_window=attention_window, hidden_dropout_prob=hidden_dropout_prob, intermediate_size=intermediate_size, hidden_act=hidden_act,
            num_hidden_layers=num_hidden_layers)
        self.pooler = LongformerPooler(hidden_size)
        self.apply(self.init_weights)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        position_ids=None,
        attention_mask=None
    ):
        r"""
        Args:
            input_ids (Tensor):
                Indices of input sequence tokens in the vocabulary. They are
                numerical representations of tokens that build the input sequence.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
            token_type_ids (Tensor, optional):
                Segment token indices to indicate first and second portions of the inputs.
                Indices can be either 0 or 1:
                - 0 corresponds to a **sentence A** token,
                - 1 corresponds to a **sentence B** token.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
                Defaults to None, which means no segment embeddings is added to token embeddings.
            position_ids (Tensor, optional):
                Indices of positions of each input sequence tokens in the position embeddings.
                Selected in the range ``[0, max_position_embeddings - 1]``.
                It's data type should be `int64` and has a shape of [batch_size, sequence_length].
                Defaults to `None`.
            attention_mask (Tensor, optional):
                Mask used in multi-head attention to avoid performing attention to some unwanted positions,
                usually the paddings or the subsequent positions.
                Its data type can be int, float and bool.
                When the data type is bool, the `masked` tokens have `False` values and the others have `True` values.
                When the data type is int, the `masked` tokens have `0` values and the others have `1` values.
                When the data type is float, the `masked` tokens have `-INF` values and the others have `0` values.
                It is a tensor with shape broadcasted to `[batch_size, num_attention_heads, sequence_length, sequence_length]`.
                For example, its shape can be  [batch_size, sequence_length], [batch_size, sequence_length, sequence_length],
                [batch_size, num_attention_heads, sequence_length, sequence_length].
                Defaults to `None`, which means nothing needed to be prevented attention to.
        Returns:
            tuple: Returns tuple (`sequence_output`, `pooled_output`).
            With the fields:
            - sequence_output (Tensor):
                Sequence of hidden-states at the last layer of the model.
                It's data type should be float32 and its shape is [batch_size, sequence_length, hidden_size].
            - pooled_output (Tensor):
                The output of first token (`[CLS]`) in sequence.
                We "pool" the model by simply taking the hidden state corresponding to the first token.
                Its data type should be float32 and its shape is [batch_size, hidden_size].
        Example:
            .. code-block::
                tokenizer = LongformerTokenizer.from_pretrained('Longformer-base-4096')
                model = LongformerModel.from_pretrained('Longformer-base-4096')
                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                sequence_output, pooled_output = model(**inputs)
        """
        if attention_mask is None:
            attention_mask = paddle.unsqueeze(
                (input_ids == self.pad_token_id
                 ).astype(self.pooler.dense.weight.dtype) * -1e9,
                axis=[1, 2])
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)
        return sequence_output, pooled_output


class LongformerForQuestionAnswering(LongformerPretrainedModel):
    r"""
    Longformer Model with a linear layer on top of the hidden-states output to compute `span_start_logits`
     and `span_end_logits`, designed for question-answering tasks like SQuAD.
    Args:
        longformer (:class:`LongformerModel`):
            An instance of LongformerModel.
        dropout (float, optional):
            The dropout probability for output of Longformer.
            If None, use the same value as `hidden_dropout_prob` of `LongformerModel`
            instance `longformer`. Defaults to `None`.
    """

    def __init__(self, longformer, dropout=None):
        super().__init__()
        self.longformer = longformer  # allow roberta to be config
        self.classifier = nn.Linear(self.longformer.config["hidden_size"], 2)
        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids=None):
        r"""
        Args:
            input_ids (Tensor):
                See :class:`LongformerModel`.
            token_type_ids (Tensor, optional):
                See :class:`LongformerModel`.
            position_ids (Tensor, optional):
                See :class:`LongformerModel`.
            attention_mask (Tensor, optional):
                See :class:`LongformerModel`.
        Returns:
            tuple: Returns tuple (`start_logits`, `end_logits`).
            With the fields:
            - `start_logits` (Tensor):
                A tensor of the input token classification logits, indicates the start position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].
            - `end_logits` (Tensor):
                A tensor of the input token classification logits, indicates the end position of the labelled span.
                Its data type should be float32 and its shape is [batch_size, sequence_length].
        Example:
            .. code-block::
                tokenizer = LongformerTokenizer.from_pretrained('longformer-base-4096')
                model = LongformerforQuestionAnswering.from_pretrained('longformer-base-4096')
                inputs = tokenizer("this is ans[sep]this is question?")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                start_logits, end_logits = model(**inputs)
        """
        sequence_output, _ = self.longformer(
            input_ids,
            token_type_ids=token_type_ids,
            position_ids=None,
            attention_mask=None)

        logits = self.classifier(sequence_output)
        logits = paddle.transpose(logits, perm=[2, 0, 1])
        start_logits, end_logits = paddle.unstack(x=logits, axis=0)

        return start_logits, end_logits
'''