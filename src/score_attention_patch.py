from __future__ import annotations

import math
import sys
from types import MethodType

import torch
import torch.nn as nn


def _maybe_update_score(past_key_values, query_states: torch.Tensor, key_states: torch.Tensor, layer_idx: int) -> None:
    if past_key_values is not None and getattr(past_key_values, "get_score", False):
        past_key_values._get_score(query_states, key_states, layer_idx)


def _maybe_apply_prune_mask(
    past_key_values,
    attention_mask: torch.Tensor | None,
    *,
    key_states: torch.Tensor,
    layer_idx: int,
    q_len: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    if past_key_values is None or not getattr(past_key_values, "pruned", False):
        return attention_mask

    prune_bias = past_key_values.build_prune_attention_bias(
        layer_idx=layer_idx,
        seq_len=key_states.shape[-2],
        q_len=q_len,
        dtype=dtype,
        device=key_states.device,
    )
    if prune_bias is None:
        return attention_mask

    if attention_mask is None:
        return prune_bias
    if attention_mask.shape[-1] != key_states.shape[-2]:
        attention_mask = attention_mask[..., : key_states.shape[-2]]
    return attention_mask + prune_bias


def _run_flash_attn_varlen(query_states, key_states, value_states, info, dropout_rate: float):
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError as exc:
        raise RuntimeError("flash_attn is required for prune_exec='gather'") from exc

    return flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=info["cu_len_q"],
        cu_seqlens_k=info["cu_len_k"],
        max_seqlen_q=info["max_len_q"],
        max_seqlen_k=info["max_len_k"],
        dropout_p=dropout_rate,
        causal=True,
    )


def _maybe_run_pruned_gather_attention(
    module,
    past_key_values,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    layer_idx: int,
    dropout_rate: float,
):
    if past_key_values is None or not getattr(past_key_values, "pruned", False):
        return None
    if getattr(past_key_values, "prune_exec", "mask") != "gather":
        return None
    if query_states.device.type != "cuda":
        raise RuntimeError("prune_exec='gather' requires CUDA tensors")

    from gather_profiler import region as _gp_region

    # v3 packed-decode fast path: q_len==1 only.
    if hasattr(past_key_values, "decode_attn"):
        with _gp_region("g0_v3_total"):
            v3_out = past_key_values.decode_attn(
                query_states, key_states, value_states, layer_idx, dropout_rate
            )
        if v3_out is not None:
            return module.o_proj(v3_out), None

    with _gp_region("g1_prepare"):
        query_states, key_states, value_states, info = past_key_values.prepare(
            query_states,
            key_states,
            value_states,
            layer_idx,
        )
    with _gp_region("g2_contig"):
        q_c = query_states.contiguous()
        k_c = key_states.contiguous()
        v_c = value_states.contiguous()
    with _gp_region("g3_kernel"):
        attn_output = _run_flash_attn_varlen(q_c, k_c, v_c, info, dropout_rate)
    with _gp_region("g4_output"):
        attn_output = attn_output.view(
            info["bsz"],
            info["n_heads_kv"],
            info["q_len"],
            info["n_group_kv"],
            info["head_dim"],
        )
        attn_output = attn_output.transpose(1, 2).reshape(info["bsz"], info["q_len"], -1).contiguous()
        out = module.o_proj(attn_output)
    return out, None


def _qwen2_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    mod = sys.modules[self.__class__.__module__]
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        _maybe_update_score(past_key_values, query_states, key_states, self.layer_idx)
        gathered_attn = _maybe_run_pruned_gather_attention(
            self,
            past_key_values,
            query_states,
            key_states,
            value_states,
            layer_idx=self.layer_idx,
            dropout_rate=0.0 if not self.training else self.attention_dropout,
        )
        if gathered_attn is not None:
            return gathered_attn
        attention_mask = _maybe_apply_prune_mask(
            past_key_values,
            attention_mask,
            key_states=key_states,
            layer_idx=self.layer_idx,
            q_len=query_states.shape[-2],
            dtype=query_states.dtype,
        )

    attention_interface = mod.eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = mod.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _qwen25_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
):
    del output_attentions, use_cache
    mod = sys.modules[self.__class__.__module__]
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, -1, self.head_dim).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = mod.apply_multimodal_rotary_pos_emb(
        query_states, key_states, cos, sin, self.rope_scaling["mrope_section"]
    )

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        _maybe_update_score(past_key_values, query_states, key_states, self.layer_idx)
        gathered_attn = _maybe_run_pruned_gather_attention(
            self,
            past_key_values,
            query_states,
            key_states,
            value_states,
            layer_idx=self.layer_idx,
            dropout_rate=0.0 if not self.training else self.attention_dropout,
        )
        if gathered_attn is not None:
            return gathered_attn
        attention_mask = _maybe_apply_prune_mask(
            past_key_values,
            attention_mask,
            key_states=key_states,
            layer_idx=self.layer_idx,
            q_len=query_states.shape[-2],
            dtype=query_states.dtype,
        )

    attention_interface = mod.eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = mod.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        position_ids=position_ids,
        **kwargs,
    )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _qwen3_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    cache_position: torch.LongTensor | None = None,
    **kwargs,
):
    mod = sys.modules[self.__class__.__module__]
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        _maybe_update_score(past_key_values, query_states, key_states, self.layer_idx)
        gathered_attn = _maybe_run_pruned_gather_attention(
            self,
            past_key_values,
            query_states,
            key_states,
            value_states,
            layer_idx=self.layer_idx,
            dropout_rate=0.0 if not self.training else self.attention_dropout,
        )
        if gathered_attn is not None:
            return gathered_attn
        attention_mask = _maybe_apply_prune_mask(
            past_key_values,
            attention_mask,
            key_states=key_states,
            layer_idx=self.layer_idx,
            q_len=query_states.shape[-2],
            dtype=query_states.dtype,
        )

    attention_interface = mod.eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = mod.ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _llava15_eager_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings=None,
    attention_mask: torch.Tensor | None = None,
    past_key_value=None,
    cache_position: torch.LongTensor | None = None,
    position_ids: torch.LongTensor | None = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    del position_ids, use_cache, kwargs
    mod = sys.modules[self.__class__.__module__]
    input_shape = hidden_states.shape[:-1]
    bsz = input_shape[0]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        _maybe_update_score(past_key_value, query_states, key_states, self.layer_idx)
        gathered_attn = _maybe_run_pruned_gather_attention(
            self,
            past_key_value,
            query_states,
            key_states,
            value_states,
            layer_idx=self.layer_idx,
            dropout_rate=0.0 if not self.training else self.attention_dropout,
        )
        if gathered_attn is not None:
            return gathered_attn[0], None, past_key_value
        attention_mask = _maybe_apply_prune_mask(
            past_key_value,
            attention_mask,
            key_states=key_states,
            layer_idx=self.layer_idx,
            q_len=query_states.shape[-2],
            dtype=query_states.dtype,
        )

    key_states = mod.repeat_kv(key_states, self.num_key_value_groups)
    value_states = mod.repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    if query_states.dtype == torch.float16:
        attn_weights = torch.where(torch.isinf(attn_weights), torch.zeros_like(attn_weights), attn_weights)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, input_shape[1], self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, input_shape[1], self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1)
    attn_output = self.o_proj(attn_output)
    if not output_attentions:
        attn_weights = None
    return attn_output, attn_weights, past_key_value


def _llava15_flash_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings=None,
):
    del position_ids, output_attentions, use_cache
    mod = sys.modules[self.__class__.__module__]
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        _maybe_update_score(past_key_value, query_states, key_states, self.layer_idx)
        gathered_attn = _maybe_run_pruned_gather_attention(
            self,
            past_key_value,
            query_states,
            key_states,
            value_states,
            layer_idx=self.layer_idx,
            dropout_rate=0.0 if not self.training else self.attention_dropout,
        )
        if gathered_attn is not None:
            return gathered_attn[0], None, past_key_value
        attention_mask = _maybe_apply_prune_mask(
            past_key_value,
            attention_mask,
            key_states=key_states,
            layer_idx=self.layer_idx,
            q_len=query_states.shape[-2],
            dtype=query_states.dtype,
        )

    key_states = mod.repeat_kv(key_states, self.num_key_value_groups)
    value_states = mod.repeat_kv(value_states, self.num_key_value_groups)
    dropout_rate = 0.0 if not self.training else self.attention_dropout

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        mod.logger.warning_once(
            "The input hidden states seems to be silently casted in float32; casting back for Flash Attention."
        )
        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window
    else:
        sliding_window = None

    attn_output = mod._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        attention_mask,
        input_shape[1],
        dropout=dropout_rate,
        sliding_window=sliding_window,
        is_causal=self.is_causal,
        use_top_left_mask=self._flash_attn_uses_top_left_mask,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def _llava15_sdpa_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: torch.LongTensor | None = None,
    position_embeddings=None,
):
    del position_ids, use_cache
    mod = sys.modules[self.__class__.__module__]
    if output_attentions:
        return _llava15_eager_forward(
            self,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = mod.apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        _maybe_update_score(past_key_value, query_states, key_states, self.layer_idx)
        gathered_attn = _maybe_run_pruned_gather_attention(
            self,
            past_key_value,
            query_states,
            key_states,
            value_states,
            layer_idx=self.layer_idx,
            dropout_rate=0.0 if not self.training else self.attention_dropout,
        )
        if gathered_attn is not None:
            return gathered_attn[0], None, past_key_value
        attention_mask = _maybe_apply_prune_mask(
            past_key_value,
            attention_mask,
            key_states=key_states,
            layer_idx=self.layer_idx,
            q_len=query_states.shape[-2],
            dtype=query_states.dtype,
        )

    key_states = mod.repeat_kv(key_states, self.num_key_value_groups)
    value_states = mod.repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    is_causal = True if causal_mask is None and input_shape[1] > 1 else False
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value


def _bind_forward(module, forward_fn) -> None:
    module.forward = MethodType(forward_fn, module)
    module._score_patch_applied = True


def apply_score_attention_patch(model) -> int:
    patch_map = {
        "Qwen2Attention": _qwen2_forward,
        "Qwen2_5_VLAttention": _qwen25_forward,
        "Qwen3VLTextAttention": _qwen3_forward,
        "LLaVAOneVision1_5_Attention": _llava15_eager_forward,
        "LLaVAOneVision1_5_FlashAttention2": _llava15_flash_forward,
        "LLaVAOneVision1_5_SdpaAttention": _llava15_sdpa_forward,
    }

    count = 0
    for module in model.modules():
        if getattr(module, "_score_patch_applied", False):
            continue
        forward_fn = patch_map.get(module.__class__.__name__)
        if forward_fn is None:
            continue
        _bind_forward(module, forward_fn)
        count += 1

    model._score_patch_count = count
    return count
