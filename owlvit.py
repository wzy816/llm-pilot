import json
from functools import lru_cache

import ftfy
import regex as re


@lru_cache()
def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class OwlViTTokenizer:
    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="<|endoftext|>",
        unk_token_id=49407,
    ):
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)

        self.decoder = {v: k for k, v in self.encoder.items()}

        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = (
                merges_handle.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
            )  # =[1:]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))

        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }

        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

        self.unk_token = unk_token
        self.unk_token_id = unk_token_id

    @property
    def vocab_size(self):
        return len(self.encoder)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        tokens = self._tokenize(text)
        return [self.encoder.get(t, self.unk_token_id) for t in tokens]

    def decode(self, ids):
        tokens = [self.decoder.get(index) for index in ids]
        return self.convert_tokens_to_string(tokens)

    def _tokenize(self, text):
        bpe_tokens = []

        text = ftfy.fix_text(text)
        text = re.sub(r"\s+", " ", text)
        text = text.strip().lower()

        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def convert_tokens_to_string(self, tokens):
        text = "".join(tokens)
        byte_array = bytearray([self.byte_decoder[c] for c in text])
        text = byte_array.decode("utf-8", errors="replace").replace("</w>", " ").strip()
        return text


from dataclasses import dataclass
from typing import *

import torch
from torch import nn


@dataclass
class OwlViTTextConfig:
    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 16  # clip是77
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    pad_token_id: int = 0
    bos_token_id: int = 49406
    eos_token_id: int = 49407


@dataclass
class OwlViTVisionConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    image_size: int = 768  # clip 是 224
    patch_size: int = 16  # 32for patch32, 16 for patch16
    layer_norm_eps: float = 1e-5
    attention_dropout: float = 0.0


@dataclass
class OwlViTConfig:
    text_config: OwlViTTextConfig = OwlViTTextConfig()
    vision_config: OwlViTVisionConfig = OwlViTVisionConfig()
    projection_dim: int = 512
    logit_scale_init_value: float = 2.6592


class OwlViTVisionEmbeddings(nn.Module):
    def __init__(self, config: OwlViTVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        patch_embeds = self.patch_embedding(
            pixel_values.to(dtype=target_dtype)
        )  # shape = [bsz, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        embeddings = embeddings + self.position_embedding(self.position_ids)
        return embeddings


class OwlViTTextEmbeddings(nn.Module):
    def __init__(self, config: OwlViTTextConfig):
        super().__init__()

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
    ) -> torch.Tensor:
        seq_length = (
            input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]
        )

        position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class OwlViTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,  # (bsz, 1, tgt_len, src_len)
        causal_attention_mask: Optional[
            torch.Tensor
        ] = None,  # (bsz, 1, tgt_len, src_len)
    ) -> torch.Tensor:
        bsz, tgt_len, embed_dim = hidden_states.size()

        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(
            query_states, key_states.transpose(1, 2)
        )  # (bsz * self.num_heads, tgt_len, src_len)

        if causal_attention_mask is not None:
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + causal_attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            attn_weights = (
                attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
                + attention_mask
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        attn_probs = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )
        attn_output = torch.bmm(
            attn_probs, value_states
        )  # (bsz * self.num_heads, tgt_len, self.head_dim)

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)
        attn_output = self.out_proj(attn_output)

        return attn_output


class QuickGELUActivation(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input * torch.sigmoid(1.702 * input)


class OwlViTMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = QuickGELUActivation()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class OwlViTEncoderLayer(nn.Module):
    def __init__(self, config: Union[OwlViTTextConfig, OwlViTVisionConfig]):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = OwlViTAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = OwlViTMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class OwlViTEncoder(nn.Module):
    def __init__(self, config: Union[OwlViTTextConfig, OwlViTVisionConfig]):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [OwlViTEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            hidden_states = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                causal_attention_mask=causal_attention_mask,
            )
        return hidden_states


class OwlViTTextModel(nn.Module):
    def __init__(self, config: OwlViTTextConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = OwlViTTextEmbeddings(config)
        self.encoder = OwlViTEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def _create_4d_causal_attention_mask(self, input_shape, dtype):
        bsz, tgt_len = input_shape
        mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min)
        mask_cond = torch.arange(mask.size(-1))
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        mask = mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len)
        return mask

    def _prepare_4d_attention_mask(self, mask, dtype):
        bsz, src_len = mask.size()
        tgt_len = src_len
        expanded_mask = (
            mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)
        )
        inverted_mask = 1.0 - expanded_mask
        return inverted_mask.masked_fill(
            inverted_mask.to(torch.bool), torch.finfo(dtype).min
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids)

        causal_attention_mask = self._create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype
        )

        attention_mask = self._prepare_4d_attention_mask(
            attention_mask, hidden_states.dtype
        )

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
        )

        last_hidden_state = self.final_layer_norm(last_hidden_state)

        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            (
                input_ids.to(dtype=torch.int, device=last_hidden_state.device)
                == self.config.eos_token_id
            )
            .int()
            .argmax(dim=-1),
        ]

        return last_hidden_state, pooled_output


class OwlViTVisionModel(nn.Module):
    def __init__(self, config: OwlViTVisionConfig):
        super().__init__()
        self.config = config

        self.embeddings = OwlViTVisionEmbeddings(config)
        self.pre_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = OwlViTEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layernorm(hidden_states)

        last_hidden_state = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            causal_attention_mask=None,
        )

        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = self.post_layernorm(pooled_output)

        return last_hidden_state, pooled_output


class OwlViTModel(nn.Module):
    def __init__(self, config: OwlViTConfig):
        super().__init__()
        self.config = config

        self.projection_dim = config.projection_dim
        self.text_embed_dim = config.text_config.hidden_size
        self.vision_embed_dim = config.vision_config.hidden_size

        self.text_model = OwlViTTextModel(config.text_config)
        self.vision_model = OwlViTVisionModel(config.vision_config)

        self.visual_projection = nn.Linear(
            self.vision_embed_dim, self.projection_dim, bias=False
        )
        self.text_projection = nn.Linear(
            self.text_embed_dim, self.projection_dim, bias=False
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(self.config.logit_scale_init_value)
        )

    def get_text_features(
        self, input_ids: torch.LongTensor, attention_mask: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        last_hidden_state, pooled_output = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_features = self.text_projection(pooled_output)

        return last_hidden_state, pooled_output, text_features

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        last_hidden_state, pooled_output = self.vision_model(
            pixel_values=pixel_values,
        )
        image_features = self.visual_projection(pooled_output)

        return last_hidden_state, pooled_output, image_features

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        image_last_hidden_states, image_pooled_output, image_embeds = (
            self.get_image_features(
                pixel_values=pixel_values,
            )
        )

        text_last_hidden_states, text_pooled_output, text_embeds = (
            self.get_text_features(
                input_ids=input_ids,
            )
        )

        image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.t()
        logits_per_text = logits_per_image.t()

        return (
            logits_per_image,
            logits_per_text,
            text_embeds,
            image_embeds,
            image_last_hidden_states,
            text_last_hidden_states,
            image_pooled_output,
            text_pooled_output,
        )


class OwlViTBoxPredictionHead(nn.Module):
    def __init__(self, config: OwlViTConfig, out_dim: int = 4):
        super().__init__()

        width = config.vision_config.hidden_size
        self.dense0 = nn.Linear(width, width)
        self.dense1 = nn.Linear(width, width)
        self.gelu = nn.GELU()
        self.dense2 = nn.Linear(width, out_dim)

    def forward(self, image_embeds: torch.Tensor) -> torch.Tensor:
        output = self.dense0(image_embeds)
        output = self.gelu(output)
        output = self.dense1(output)
        output = self.gelu(output)
        output = self.dense2(output)
        return output


class OwlViTClassPredictionHead(nn.Module):
    def __init__(self, config: OwlViTConfig):
        super().__init__()

        self.query_dim = config.vision_config.hidden_size

        self.dense0 = nn.Linear(self.query_dim, config.text_config.hidden_size)
        self.logit_shift = nn.Linear(self.query_dim, 1)
        self.logit_scale = nn.Linear(self.query_dim, 1)
        self.elu = nn.ELU()

    def forward(
        self,
        image_embeds: torch.Tensor,
        query_embeds: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        image_class_embeds = self.dense0(image_embeds)

        image_class_embeds = image_class_embeds / (
            torch.linalg.norm(image_class_embeds, dim=-1, keepdim=True) + 1e-6
        )
        query_embeds = query_embeds / (
            torch.linalg.norm(query_embeds, dim=-1, keepdim=True) + 1e-6
        )
        pred_logits = torch.einsum(
            "...pd,...qd->...pq", image_class_embeds, query_embeds
        )

        logit_shift = self.logit_shift(image_embeds)
        logit_scale = self.logit_scale(image_embeds)
        logit_scale = self.elu(logit_scale) + 1
        pred_logits = (pred_logits + logit_shift) * logit_scale

        return (pred_logits, image_class_embeds)


class OwlViTForObjectDetection(nn.Module):
    def __init__(self, config: OwlViTConfig):
        super().__init__()
        self.owlvit = OwlViTModel(config)
        self.class_head = OwlViTClassPredictionHead(config)
        self.box_head = OwlViTBoxPredictionHead(config)

        self.layer_norm = nn.LayerNorm(
            config.vision_config.hidden_size, eps=config.vision_config.layer_norm_eps
        )
        self.sigmoid = nn.Sigmoid()

        self.sqrt_num_patches = (
            config.vision_config.image_size // config.vision_config.patch_size
        )

    def compute_box_bias(self, feature_map: torch.Tensor) -> torch.Tensor:
        num_patches = feature_map.shape[1]
        box_coordinates = torch.cartesian_prod(
            torch.arange(1, num_patches + 1), torch.arange(1, num_patches + 1)
        )
        box_coordinates = box_coordinates.reshape(
            num_patches, num_patches, -1
        ).transpose(0, 1)
        box_coordinates = box_coordinates / num_patches
        box_coordinates = box_coordinates.reshape(-1, 2).to(feature_map.device)

        box_coordinates = torch.clip(box_coordinates, 0.0, 1.0)

        # Unnormalize xy
        box_coord_bias = torch.log(box_coordinates + 1e-4) - torch.log1p(
            -box_coordinates + 1e-4
        )

        box_size = torch.full_like(box_coord_bias, 1.0 / feature_map.shape[-2])
        box_size_bias = torch.log(box_size + 1e-4) - torch.log1p(-box_size + 1e-4)

        # Compute box bias
        box_bias = torch.cat([box_coord_bias, box_size_bias], dim=-1)
        return box_bias

    def box_predictor(
        self,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        pred_boxes = self.box_head(image_embeds)  # (batch_size, num_boxes, 4)

        feature_map = image_embeds.reshape(
            image_embeds.shape[0],
            self.sqrt_num_patches,
            self.sqrt_num_patches,
            image_embeds.shape[-1],
        )

        pred_boxes += self.compute_box_bias(feature_map)
        pred_boxes = self.sigmoid(pred_boxes)
        return pred_boxes

    def text_embedder(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        _, _, text_embeds = self.owlvit.get_text_features(
            input_ids=input_ids, attention_mask=attention_mask
        )
        return text_embeds

    def image_embedder(
        self,
        pixel_values: torch.FloatTensor,
    ) -> torch.Tensor:
        # norm on non-projected output
        image_last_hidden_states, _ = self.owlvit.vision_model(
            pixel_values=pixel_values
        )
        image_embeds = self.owlvit.vision_model.post_layernorm(image_last_hidden_states)

        # Merge
        class_token_out = torch.broadcast_to(
            image_embeds[:, :1, :], image_embeds[:, :-1].shape
        )
        image_embeds = image_embeds[:, 1:, :] * class_token_out
        image_embeds = self.layer_norm(image_embeds)

        return image_embeds

    def embed_image_query(
        self,
        query_image_features: torch.Tensor,
    ) -> torch.Tensor:

        class_embeds = self.class_head.dense0(query_image_features)

        pred_boxes = self.box_predictor(query_image_features)
        pred_boxes_as_corners = self._center_to_corners_format_torch(pred_boxes)

        # Loop over query images
        best_class_embeds = []
        best_box_indices = []
        pred_boxes_device = pred_boxes_as_corners.device

        for i in range(query_image_features.shape[0]):
            each_query_box = torch.tensor([[0, 0, 1, 1]], device=pred_boxes_device)
            each_query_pred_boxes = pred_boxes_as_corners[i]
            ious, _ = box_iou(each_query_box, each_query_pred_boxes)

            if torch.all(ious[0] == 0.0):
                ious = generalized_box_iou(each_query_box, each_query_pred_boxes)

            iou_threshold = torch.max(ious) * 0.8

            selected_inds = (ious[0] >= iou_threshold).nonzero()
            if selected_inds.numel():
                selected_embeddings = class_embeds[i][selected_inds.squeeze(1)]
                mean_embeds = torch.mean(class_embeds[i], axis=0)
                mean_sim = torch.einsum("d,id->i", mean_embeds, selected_embeddings)
                best_box_ind = selected_inds[torch.argmin(mean_sim)]
                best_class_embeds.append(class_embeds[i][best_box_ind])
                best_box_indices.append(best_box_ind)
        if best_class_embeds:
            query_embeds = torch.stack(best_class_embeds)
            box_indices = torch.stack(best_box_indices)
        else:
            query_embeds, box_indices = None, None

        return query_embeds, box_indices, pred_boxes

    def _center_to_corners_format_torch(
        self, bboxes_center: torch.Tensor
    ) -> torch.Tensor:
        center_x, center_y, width, height = bboxes_center.unbind(-1)
        bbox_corners = torch.stack(
            [
                (center_x - 0.5 * width),
                (center_y - 0.5 * height),
                (center_x + 0.5 * width),
                (center_y + 0.5 * height),
            ],
            dim=-1,
        )
        return bbox_corners

    def text_guided_detection(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        pixel_values: torch.FloatTensor,
    ):

        text_embeds = self.text_embedder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        query_embeds = text_embeds
        image_embeds = self.image_embedder(pixel_values=pixel_values)

        pred_logits, class_embeds = self.class_head(image_embeds, query_embeds)

        pred_boxes = self.box_predictor(image_embeds)
        pred_boxes = self._center_to_corners_format_torch(pred_boxes)

        return (
            text_embeds,
            None,
            image_embeds,
            pred_logits,
            class_embeds,
            pred_boxes,
        )

    def image_guided_detection(
        self,
        query_pixel_values: torch.FloatTensor,
        pixel_values: torch.FloatTensor,
    ):

        # embed
        image_embeds = self.image_embedder(pixel_values=pixel_values)

        query_image_embeds = self.image_embedder(pixel_values=query_pixel_values)
        query_embeds, _, _ = self.embed_image_query(query_image_embeds)

        # enable multi query on multi images (not recommended though)
        query_embeds = torch.squeeze(query_embeds, 1)

        pred_logits, class_embeds = self.class_head(image_embeds, query_embeds)

        pred_boxes = self.box_predictor(image_embeds)
        pred_boxes = self._center_to_corners_format_torch(pred_boxes)

        return (
            None,
            query_image_embeds,
            image_embeds,
            pred_logits,
            class_embeds,
            pred_boxes,
        )


def box_iou(boxes1, boxes2):
    def _upcast(t: torch.Tensor) -> torch.Tensor:
        if t.is_floating_point():
            return t if t.dtype in (torch.float32, torch.float64) else t.float()
        else:
            return t if t.dtype in (torch.int32, torch.int64) else t.int()

    def box_area(boxes: torch.Tensor) -> torch.Tensor:
        boxes = _upcast(boxes)
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
    inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):

    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(
            f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}"
        )
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(
            f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}"
        )
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


import gc
import glob
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_TYPE = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
DEVICE = "cuda"


def cal_line_text(w, h):
    line_width = int(min(10, min(w, h) / 100))
    font_size = int(max(10, min(w, h) / 40))
    return 4, 4
    return line_width, font_size


def multi_text_on_multi_image(
    model,
    input_ids,
    query_texts,
    image_names,
    image_arrs,
    image_scales,
    image_ios,
    output_dir,
):

    with torch.inference_mode(), torch.no_grad():
        input_ids = torch.LongTensor(input_ids).to(DEVICE)
        attention_mask = input_ids.ne(0).to(
            DEVICE
        )  # model_config.text_config.pad_token_id
        pixel_values = torch.Tensor(np.array(image_arrs)).to(DEVICE)

        (
            text_embeds,
            _,
            image_embeds,
            text_pred_logits,
            text_class_embeds,
            text_pred_boxes,
        ) = model.text_guided_detection(input_ids, attention_mask, pixel_values)

        text_probs = torch.max(text_pred_logits, dim=-1)
        text_scores = torch.sigmoid(text_probs.values)
        text_labels = text_probs.indices

        text_boxes = text_pred_boxes

        text_box_areas = (text_boxes[:, :, 2] - text_boxes[:, :, 0]) * (
            text_boxes[:, :, 3] - text_boxes[:, :, 1]
        )

        nms_threshold = 0.1
        for idx in range(text_boxes.shape[0]):
            for i in torch.argsort(-text_scores[idx]):
                if not text_scores[idx][i]:
                    continue
                ious = box_iou(text_boxes[idx][i, :].unsqueeze(0), text_boxes[idx])[0][
                    0
                ]
                ious[i] = -1.0
                text_scores[idx][ious > nms_threshold] = 0.0

        scale_fct = torch.Tensor(image_scales).to(text_boxes.device)
        text_boxes = text_boxes * scale_fct[:, None, :]

    text_score_threshold = 0.01
    text_area_threshold = 0.1
    text_results = []

    for s, l, b, ar in zip(text_scores, text_labels, text_boxes, text_box_areas):
        score = s[torch.logical_and(s > text_score_threshold, ar < text_area_threshold)]
        label = l[torch.logical_and(s > text_score_threshold, ar < text_area_threshold)]
        box = b[torch.logical_and(s > text_score_threshold, ar < text_area_threshold)]
        text_results.append({"scores": score, "labels": label, "boxes": box})

    # plot text od bbox
    for result, image_io, image_scale, image_name in zip(
        text_results, image_ios, image_scales, image_names
    ):
        im = image_io.copy()
        draw = ImageDraw.Draw(im)

        w, h, _, _ = image_scale
        line_width, text_size = cal_line_text(w, h)

        boxes, scores, labels = result["boxes"], result["scores"], result["labels"]
        for box, score, label in zip(boxes, scores, labels):
            font = ImageFont.truetype(FONT_TYPE, text_size)
            xmin, ymin, xmax, ymax = box.tolist()
            import math

            xmin = math.floor(xmin)
            xmax = math.ceil(xmax)
            ymin = math.floor(ymin)
            ymax = math.ceil(ymax)

            draw.rectangle(
                (xmin, ymin, xmax, ymax),
                outline="red",
                width=line_width,
            )
            t = query_texts[label]
            draw.text(
                (xmin, ymin - text_size),
                f"{t}:{round(score.item(), 3)}",
                fill="red",
                font=font,
            )

        out = os.path.join(output_dir, os.path.basename(image_name))
        im.save(out)


def one_image_on_multi_image(
    model,
    query_image_ids,
    query_image_names,
    query_image_arrs,
    image_ids,
    image_names,
    image_arrs,
    image_scales,
    image_ios,
):

    image_results = []

    for query_image_id, query_image in zip(query_image_ids, query_image_arrs):

        with torch.inference_mode(), torch.no_grad():

            query_pixel_values = torch.Tensor(np.array([query_image])).to(DEVICE)
            pixel_values = torch.Tensor(np.array(image_arrs)).to(DEVICE)

            (
                _,
                query_image_embeds,
                image_embeds,
                image_pred_logits,
                image_class_embeds,
                image_pred_boxes,
            ) = model.image_guided_detection(query_pixel_values, pixel_values)

            image_probs = torch.max(image_pred_logits, dim=-1)
            image_scores = torch.sigmoid(image_probs.values)  # (bsz, 576)
            image_labels = image_probs.indices  # (bsz, 576)

            image_boxes = image_pred_boxes  # (bsz, 576, 4)

            image_box_areas = (image_boxes[:, :, 2] - image_boxes[:, :, 0]) * (
                image_boxes[:, :, 3] - image_boxes[:, :, 1]
            )

            image_nms_threshold = 0.3
            for idx in range(image_boxes.shape[0]):
                for i in torch.argsort(-image_scores[idx]):
                    if not image_scores[idx][i]:
                        continue
                    ious = box_iou(
                        image_boxes[idx][i, :].unsqueeze(0), image_boxes[idx]
                    )[0][0]
                    ious[i] = -1.0
                    image_scores[idx][ious > image_nms_threshold] = 0.0

            scale_fct = torch.Tensor(image_scales).to(image_boxes.device)
            image_boxes = image_boxes * scale_fct[:, None, :]

        image_score_threshold = 0.8
        image_area_threshold = 0.9

        for s, l, b, ar, image_id in zip(
            image_scores, image_labels, image_boxes, image_box_areas, image_ids
        ):
            score = s[
                torch.logical_and(s > image_score_threshold, ar < image_area_threshold)
            ]
            #         label = l[s > image_threshold]
            box = b[
                torch.logical_and(s > image_score_threshold, ar < image_area_threshold)
            ]
            if score.shape[0] > 0:
                image_results.append(
                    {
                        "scores": score,
                        "boxes": box,
                        "query_image_id": query_image_id,
                        "image_id": image_id,
                    }
                )

    for image_io, image_id, image_scale, image_name in zip(
        image_ios, image_ids, image_scales, image_names
    ):
        results = [r for r in image_results if r["image_id"] == image_id]

        if len(results) == 0:
            continue

        im = image_io.copy()
        draw = ImageDraw.Draw(im)

        w, h, _, _ = image_scale
        line_width, text_size = cal_line_text(w, h)

        for result in results:
            boxes, scores = result["boxes"], result["scores"]
            query_image_id = result["query_image_id"]

            for box, score in zip(boxes, scores):
                font = ImageFont.truetype(FONT_TYPE, text_size)
                xmin, ymin, xmax, ymax = box.tolist()
                import math

                xmin = math.floor(xmin)
                xmax = math.ceil(xmax)
                ymin = math.floor(ymin)
                ymax = math.ceil(ymax)

                draw.rectangle(
                    (xmin, ymin, xmax, ymax), outline="green", width=line_width
                )

                t = query_image_names[query_image_id]
                draw.text(
                    (xmin, ymin + 10),
                    f"{t}:{round(score.item(), 3)}",
                    fill="white",
                    font=font,
                )

        im.save(Path(image_name).with_suffix(".image_on_image.jpg"))
        # display(im)


"""
pip install ftfy

CUDA_VISIBLE_DEVICES=1,2,3 python3 owlvit.py

"""


def main():
    torch.set_default_device(DEVICE)

    # torch.set_default_dtype(torch.float32)  # patch32
    torch.set_default_dtype(torch.bfloat16)  # patch16

    # load model
    model_dir = "/mnt/.cache/models--google--owlvit-base-patch16/snapshots/4b420debb9c806fc4caf9ecc8efb72208c0db892/"
    model_path = model_dir + "pytorch_model.bin"
    vocab_path = model_dir + "vocab.json"
    merges_path = model_dir + "merges.txt"

    # state_dict = torch.load(model_path, map_location=DEVICE)
    state_dict = torch.load(model_path)

    model_config = OwlViTConfig()
    model = OwlViTForObjectDetection(model_config)
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)

    model_params = set([name for name, param in model.named_parameters()])
    for key in state_dict.keys():
        if key not in model_params:
            print(key)
    del state_dict
    gc.collect()

    RESIZE_IMAGE_SIZE = model_config.vision_config.image_size
    TEXT_CONFIG = model_config.text_config

    def process_image(image):
        image_mean = [0.48145466, 0.4578275, 0.40821073]
        image_std = [0.26862954, 0.26130258, 0.27577711]
        resized_image = image.convert("RGB").resize(
            (RESIZE_IMAGE_SIZE, RESIZE_IMAGE_SIZE),
            resample=Image.Resampling.BICUBIC,
        )
        image_arr = np.array(resized_image)
        image_arr = image_arr * 1 / 225.0
        normalized_image_arr = (image_arr - image_mean) / image_std
        normalized_image_arr = normalized_image_arr.transpose(2, 0, 1)
        return normalized_image_arr

    # load tokenizer
    tokenizer = OwlViTTokenizer(vocab_path, merges_path)

    def encode_text(text):
        tokens = tokenizer.encode(text)
        tokens = [TEXT_CONFIG.bos_token_id] + tokens + [TEXT_CONFIG.eos_token_id]
        if len(tokens) < TEXT_CONFIG.max_position_embeddings:
            tokens = tokens + (TEXT_CONFIG.max_position_embeddings - len(tokens)) * [
                TEXT_CONFIG.pad_token_id
            ]
        return tokens

    # load images
    image_dir = "./data/rack"
    image_ids = []
    image_names = []
    image_ios = []
    image_scales = []
    image_formats = []
    image_arrs = []
    for image_id, image_name in enumerate(glob.glob(f"{image_dir}/*.jpg")):
        image_ids.append(image_id)
        image_names.append(image_name)
        image = Image.open(image_name)
        image_ios.append(image)

        w, h = image.size
        image_scales.append([w, h, w, h])

        image_formats.append(image.format)

        image_arrs.append(process_image(image))

    # inference
    query_texts = [
        "a drink",
        "a beverage",
        "A liquid in a container",
        "A bottle of drink",
        "A can of soda",
        "A bottle of water",
    ]
    input_ids = [encode_text(t) for t in query_texts]
    output_dir = "./data/rack_texts_on_image"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    multi_text_on_multi_image(
        model,
        input_ids,
        query_texts,
        image_names,
        image_arrs,
        image_scales,
        image_ios,
        output_dir,
    )

    # inference

    # query_image_name = "./data/"
    # query_image_names = [query_image_name]
    # query_image_ids = [0]
    # query_image = Image.open(query_image_name)
    # query_image_arrs = [process_image(query_image)]

    # one_image_on_multi_image(
    #     model,
    #     query_image_ids,
    #     query_image_names,
    #     query_image_arrs,
    #     image_ids,
    #     image_names,
    #     image_arrs,
    #     image_scales,
    #     image_ios,
    # )


if __name__ == "__main__":
    main()
