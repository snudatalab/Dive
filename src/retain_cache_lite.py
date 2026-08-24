from __future__ import annotations

from collections.abc import Iterable
import math
import os
from typing import Any

import torch
from transformers.cache_utils import DynamicCache

_GATHER_FASTPATH_ENABLED = os.environ.get("KVZIP_GATHER_FASTPATH", "1") != "0"
_GATHER_PACKED_ENABLED = os.environ.get("KVZIP_GATHER_PACKED", "1") != "0"
_GATHER_PACKED_MAX_SUFFIX = int(os.environ.get("KVZIP_GATHER_PACKED_MAX_SUFFIX", "256"))


class RetainCacheLite(DynamicCache):
    """Minimal custom cache for KVzip-style prefill/reuse/rollback experiments.

    This deliberately stays close to Hugging Face `DynamicCache` and only adds:
    - lightweight metadata for prefill bookkeeping
    - an explicit `slice()` rollback API like KVzip

    Keeping this class thin reduces the risk of diverging from model-specific
    transformer forward logic while still giving us a real custom cache object
    to extend later with scoring/pruning behavior.
    """

    def __init__(
        self,
        ddp_cache_data: Iterable[tuple[torch.Tensor, torch.Tensor]] | None = None,
        config=None,
        offloading: bool = False,
        offload_only_non_sliding: bool = False,
    ):
        super().__init__(
            ddp_cache_data=ddp_cache_data,
            config=config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )
        self.prefill_ids: torch.Tensor | None = None
        self.ctx_ids: torch.Tensor | None = None
        self.prefill_len: int = 0
        self.get_score: bool = False
        self.score: list[torch.Tensor | None] = []
        self.ctx_start: int = 0
        self.ctx_end: int = 0
        self.score_query_start: int = 0
        self.n_layers: int = self._resolve_num_hidden_layers(config)
        self.n_heads: int = self._resolve_num_attention_heads(config)
        self.n_heads_kv: int = self._resolve_num_key_value_heads(config)
        self.n_group_kv: int = self._resolve_num_query_groups()
        self.valid: torch.Tensor | None = None
        self.pruned: bool = False
        self.prune_mode: str | None = None
        self.prune_exec: str = "mask"
        self.requested_prune_ratio: float = 1.0
        self.actual_prune_ratio: float = 1.0
        self.prune_diagnostics: dict[str, Any] | None = None
        self.frame_count_for_prune: int | None = None
        self.adaptive_alpha_for_prune: float = 0.5
        self.adaptive_min_keep_per_frame: int = 1
        self.adaptive_entropy_eps: float = 1e-6
        self.layer_early_ratio_gain_for_prune: float = 1.0
        self.token_select_mode_for_prune: str = "deterministic"
        self.token_select_temperature_for_prune: float = 1.0
        self.token_select_noise_scale_for_prune: float = 1.0
        self.token_select_seed_for_prune: int | None = None
        self.head_stat_alpha_for_prune: float = 1.0
        self.head_stat_beta_for_prune: float = 1.0
        self.head_stat_eps_for_prune: float = 1e-8
        self.head_stat_use_layer_norm_for_prune: bool = False
        self.head_stat_mode_for_prune: str = "meanstd"
        self.head_stat_min_uniform_ratio_for_prune: float = 0.0
        self.head_select_mode_for_prune: str = "greedy"
        self.head_div_candidate_ratio_for_prune: float = 2.0
        self.head_div_lambda_for_prune: float = 0.25
        self.head_div_eps_for_prune: float = 1e-6
        self.head_continuous_opt_steps_for_prune: int = 0
        self.head_continuous_step_size_for_prune: float = 0.1
        self.head_continuous_init_temp_for_prune: float = 1.0
        self.head_continuous_final_temp_for_prune: float = 0.1
        self.head_dpp_score_alpha_for_prune: float = 4.0
        self.head_dpp_diag_jitter_for_prune: float = 1e-6
        self.head_dpp_enable_diagnostics_for_prune: bool = True
        self.debug_score: bool = False
        self.debug_score_layer: int = 0
        self.debug_score_record: dict[str, Any] | None = None
        self.score_aggregation_mode: str = "max"

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def set_prefill_metadata(
        self,
        *,
        prefill_ids: torch.Tensor | None = None,
        ctx_ids: torch.Tensor | None = None,
    ) -> None:
        if prefill_ids is not None:
            self.prefill_ids = prefill_ids.detach().clone()
            self.prefill_len = int(prefill_ids.shape[-1])
        if ctx_ids is not None:
            self.ctx_ids = ctx_ids.detach().clone()

    def slice(self, seen_token_prev: int) -> None:
        if seen_token_prev < 0:
            raise ValueError("seen_token_prev must be non-negative")
        self.crop(seen_token_prev)

    @property
    def ctx_len(self) -> int:
        return max(self.ctx_end - self.ctx_start, 0)

    def init_score(
        self,
        *,
        ctx_start: int = 0,
        ctx_end: int | None = None,
        query_start: int = 0,
        debug_capture: bool = False,
        debug_layer_idx: int = 0,
    ) -> None:
        self.ctx_start = ctx_start
        self.ctx_end = self.prefill_len if ctx_end is None else ctx_end
        self.score_query_start = max(query_start, 0)
        self.get_score = True
        self.valid = None
        self.pruned = False
        self.prune_mode = None
        self.prune_exec = "mask"
        self.requested_prune_ratio = 1.0
        self.actual_prune_ratio = 1.0
        self.prune_diagnostics = None
        num_layers = max(self.n_layers, len(self))
        self.score = [None for _ in range(num_layers)]
        self.debug_score = debug_capture
        self.debug_score_layer = debug_layer_idx
        self.debug_score_record = None

    def set_frame_layout_for_prune(self, *, frame_count: int | None) -> None:
        if frame_count is None:
            self.frame_count_for_prune = None
            return
        self.frame_count_for_prune = int(frame_count) if int(frame_count) > 0 else None

    def set_adaptive_prune_config(
        self,
        *,
        alpha: float | None = None,
        min_keep_per_frame: int | None = None,
        entropy_eps: float | None = None,
        token_select_mode: str | None = None,
        token_select_temperature: float | None = None,
        token_select_noise_scale: float | None = None,
        token_select_seed: int | None = None,
    ) -> None:
        if alpha is not None:
            self.adaptive_alpha_for_prune = float(min(max(alpha, 0.0), 1.0))
        if min_keep_per_frame is not None:
            self.adaptive_min_keep_per_frame = max(int(min_keep_per_frame), 0)
        if entropy_eps is not None:
            self.adaptive_entropy_eps = max(float(entropy_eps), 1e-12)
        if token_select_mode is not None:
            token_select_mode_norm = str(token_select_mode).strip().lower()
            if token_select_mode_norm not in {"deterministic", "gumbel_topk"}:
                raise ValueError(f"Unsupported token_select_mode: {token_select_mode}")
            self.token_select_mode_for_prune = token_select_mode_norm
        if token_select_temperature is not None:
            self.token_select_temperature_for_prune = max(float(token_select_temperature), 1e-6)
        if token_select_noise_scale is not None:
            self.token_select_noise_scale_for_prune = max(float(token_select_noise_scale), 0.0)
        if token_select_seed is not None:
            self.token_select_seed_for_prune = int(token_select_seed)

    def set_score_aggregation_mode(self, mode: str) -> None:
        mode_norm = str(mode).strip().lower()
        if mode_norm not in {"max", "sum_headmax"}:
            raise ValueError(f"Unsupported score aggregation mode: {mode}")
        self.score_aggregation_mode = mode_norm

    def set_layer_linear_prune_config(self, *, early_ratio_gain: float | None = None) -> None:
        if early_ratio_gain is not None:
            self.layer_early_ratio_gain_for_prune = max(float(early_ratio_gain), 1e-6)

    def set_headwise_stat_prune_config(
        self,
        *,
        alpha: float | None = None,
        beta: float | None = None,
        eps: float | None = None,
        use_layer_norm: bool | None = None,
        stat_mode: str | None = None,
        min_uniform_ratio: float | None = None,
    ) -> None:
        if alpha is not None:
            self.head_stat_alpha_for_prune = float(alpha)
        if beta is not None:
            self.head_stat_beta_for_prune = float(beta)
        if eps is not None:
            self.head_stat_eps_for_prune = max(float(eps), 1e-12)
        if use_layer_norm is not None:
            self.head_stat_use_layer_norm_for_prune = bool(use_layer_norm)
        if stat_mode is not None:
            mode_norm = str(stat_mode).strip().lower()
            if mode_norm not in {"mean", "std", "meanstd"}:
                raise ValueError(f"Unsupported head stat mode: {stat_mode}")
            self.head_stat_mode_for_prune = mode_norm
        if min_uniform_ratio is not None:
            self.head_stat_min_uniform_ratio_for_prune = float(
                min(max(float(min_uniform_ratio), 0.0), 1.0)
            )

    def set_headwise_diversity_prune_config(
        self,
        *,
        candidate_ratio: float | None = None,
        diversity_lambda: float | None = None,
        eps: float | None = None,
    ) -> None:
        if candidate_ratio is not None:
            self.head_div_candidate_ratio_for_prune = max(float(candidate_ratio), 1.0)
        if diversity_lambda is not None:
            self.head_div_lambda_for_prune = max(float(diversity_lambda), 0.0)
        if eps is not None:
            self.head_div_eps_for_prune = max(float(eps), 1e-12)

    def set_headwise_select_prune_config(
        self,
        *,
        select_mode: str | None = None,
    ) -> None:
        if select_mode is None:
            return

        mode_norm = str(select_mode).strip().lower()
        alias_map = {
            "greedy": "greedy",
            "default": "greedy",
            "local_head_continuous": "local_head_continuous",
            "continuous": "local_head_continuous",
            "local_head_dpp": "local_head_dpp",
            "dpp": "local_head_dpp",
            "layer_global_dpp_quota": "layer_global_dpp_quota",
            "global_dpp": "layer_global_dpp_quota",
            "layer_dpp": "layer_global_dpp_quota",
        }
        if mode_norm not in alias_map:
            raise ValueError(f"Unsupported head_select_mode: {select_mode}")
        self.head_select_mode_for_prune = alias_map[mode_norm]

    def set_headwise_dpp_prune_config(
        self,
        *,
        score_alpha: float | None = None,
        diag_jitter: float | None = None,
        enable_diagnostics: bool | None = None,
    ) -> None:
        if score_alpha is not None:
            self.head_dpp_score_alpha_for_prune = float(score_alpha)
        if diag_jitter is not None:
            self.head_dpp_diag_jitter_for_prune = max(float(diag_jitter), 1e-12)
        if enable_diagnostics is not None:
            self.head_dpp_enable_diagnostics_for_prune = bool(enable_diagnostics)

    def set_headwise_continuous_prune_config(
        self,
        *,
        opt_steps: int | None = None,
        step_size: float | None = None,
        init_temp: float | None = None,
        final_temp: float | None = None,
        entropy_reg: float | None = None,
    ) -> None:
        if opt_steps is not None:
            self.head_continuous_opt_steps_for_prune = max(int(opt_steps), 0)
        if step_size is not None:
            self.head_continuous_step_size_for_prune = max(float(step_size), 1e-8)
        if init_temp is not None:
            self.head_continuous_init_temp_for_prune = max(float(init_temp), 1e-6)
        if final_temp is not None:
            self.head_continuous_final_temp_for_prune = max(float(final_temp), 1e-6)
        _ = entropy_reg

    def _update_score(self, layer_idx: int, score: torch.Tensor) -> None:
        if layer_idx >= len(self.score):
            self.score.extend([None] * (layer_idx + 1 - len(self.score)))
        current = self.score[layer_idx]
        if current is None:
            self.score[layer_idx] = score.detach()
        else:
            self.score[layer_idx] = torch.maximum(current, score.detach())

    def _resolve_uniform_head_select_mode(self) -> str:
        mode_norm = str(self.head_select_mode_for_prune).strip().lower()
        if mode_norm in {"", "greedy", "default"}:
            return "greedy"
        if mode_norm in {"local_head_continuous", "continuous"}:
            return "local_head_continuous"
        if mode_norm in {"local_head_dpp", "dpp"}:
            return "local_head_dpp"
        if mode_norm in {"layer_global_dpp_quota", "global_dpp", "layer_dpp"}:
            return "layer_global_dpp_quota"
        raise ValueError(f"Unsupported head_select_mode: {self.head_select_mode_for_prune}")

    def _get_score(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        layer_idx: int,
    ) -> None:
        if not self.get_score or self.ctx_len <= 0:
            return

        query_start = min(self.score_query_start, query_states.shape[-2])
        if query_start >= query_states.shape[-2]:
            return

        query_states = query_states[:, :, query_start:, :]
        key_states = key_states[:, :, self.ctx_start:self.ctx_end, :]
        if key_states.shape[-2] == 0:
            return

        bsz, num_heads, q_len, head_dim = query_states.shape
        num_kv_heads = key_states.shape[1]
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"query heads ({num_heads}) must be divisible by kv heads ({num_kv_heads})"
            )

        query_states = query_states.view(bsz, num_kv_heads, -1, q_len, head_dim)
        key_states = key_states.unsqueeze(2).transpose(-2, -1).contiguous()

        attn_weights = torch.matmul(query_states.float(), key_states.float()) / math.sqrt(head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        if self.score_aggregation_mode == "sum_headmax":
            score = attn_weights.amax(dim=-2).sum(dim=-2).to(query_states.dtype)
        else:
            score = attn_weights.amax(dim=(-3, -2)).to(query_states.dtype)
        if self.debug_score and layer_idx == self.debug_score_layer and self.debug_score_record is None:
            self.debug_score_record = {
                "layer_idx": layer_idx,
                "query_states": query_states.detach().cpu().clone(),
                "key_states": key_states.detach().cpu().clone(),
                "score": score.detach().cpu().clone(),
                "score_aggregation_mode": self.score_aggregation_mode,
            }
        self._update_score(layer_idx, score)

    @staticmethod
    def manual_score_from_debug_record(debug_record: dict[str, Any]) -> torch.Tensor:
        query_states = debug_record["query_states"].float()
        key_states = debug_record["key_states"].float()
        score_aggregation_mode = str(debug_record.get("score_aggregation_mode", "max")).strip().lower()

        head_dim = query_states.shape[-1]
        attn_weights = torch.matmul(query_states, key_states) / math.sqrt(head_dim)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        if score_aggregation_mode == "sum_headmax":
            return attn_weights.amax(dim=-2).sum(dim=-2).to(debug_record["score"].dtype)
        return attn_weights.amax(dim=(-3, -2)).to(debug_record["score"].dtype)

    def prune(
        self,
        ratio: float,
        mode: str = "global",
        exec_mode: str = "mask",
    ) -> dict[str, float | int | str]:
        if not 0 < ratio <= 1:
            raise ValueError("ratio must be in (0, 1]")
        if mode not in {
            "global",
            "uniform",
            "uniform_per_frame",
            "adaptive_per_frame",
            "uniform_layer_linear",
        }:
            raise ValueError(f"Unsupported prune mode: {mode}")
        if exec_mode not in {"mask", "gather"}:
            raise ValueError(f"Unsupported prune exec mode: {exec_mode}")
        if self.ctx_len <= 0:
            raise ValueError("ctx_len must be positive before pruning")

        scores = self._stack_scores()
        if mode == "global":
            valid, threshold = self._threshold_global(scores, ratio)
        elif mode == "uniform_per_frame":
            valid, threshold = self._threshold_uniform_per_frame(scores, ratio)
        elif mode == "adaptive_per_frame":
            valid, threshold = self._threshold_adaptive_per_frame(scores, ratio)
        elif mode == "uniform_layer_linear":
            valid, threshold = self._threshold_uniform_layer_linear(scores, ratio)
        else:
            valid, threshold = self._threshold_uniform(scores, ratio)

        self.valid = valid
        self.pruned = True
        self.prune_mode = mode
        self.prune_exec = exec_mode
        self.requested_prune_ratio = float(ratio)
        self.actual_prune_ratio = float(valid.float().mean().item())
        # Invalidate gather fastpath caches; will be rebuilt lazily per layer.
        self._gather_fastpath_cache = {}
        # Layer-shared per-suf_len step-layout cache (head_id_suf, suf_off, ...).
        self._gather_step_layout_cache = {}
        # v3 incremental packed K/V buffers, per (layer_idx, device).
        self._gather_packed_cache = {}
        result = {
            "mode": mode,
            "exec": exec_mode,
            "requested_ratio": float(ratio),
            "actual_ratio": self.actual_prune_ratio,
            "threshold": float(threshold),
            "ctx_len": int(self.ctx_len),
            "layers": int(valid.shape[0]),
            "kv_heads": int(valid.shape[1]),
        }
        if self.prune_diagnostics is not None:
            selector = str(self.prune_diagnostics.get("selector", "")).strip().lower()
            if selector == "continuous":
                result["continuous_optimized_heads"] = int(
                    self.prune_diagnostics.get("optimized_heads", 0)
                )
                result["continuous_fallback_heads"] = int(
                    self.prune_diagnostics.get("fallback_heads", 0)
                )
            elif selector == "dpp":
                result["dpp_total_heads"] = int(self.prune_diagnostics.get("total_heads", 0))
                result["dpp_changed_heads"] = int(self.prune_diagnostics.get("changed_heads", 0))
                result["dpp_identical_heads"] = int(self.prune_diagnostics.get("identical_heads", 0))
                result["dpp_mean_overlap_ratio"] = float(
                    self.prune_diagnostics.get("mean_overlap_ratio", 0.0)
                )
                result["dpp_mean_selected_pairwise_sim"] = float(
                    self.prune_diagnostics.get("mean_selected_pairwise_sim", 0.0)
                )
                result["dpp_mean_greedy_pairwise_sim"] = float(
                    self.prune_diagnostics.get("mean_greedy_pairwise_sim", 0.0)
                )
                result["dpp_mean_score_ratio_vs_greedy"] = float(
                    self.prune_diagnostics.get("mean_score_ratio_vs_greedy", 0.0)
                )
            elif selector == "layer_global_dpp_quota":
                result["layer_global_dpp_total_heads"] = int(
                    self.prune_diagnostics.get("total_heads", 0)
                )
                result["layer_global_dpp_changed_heads"] = int(
                    self.prune_diagnostics.get("changed_heads", 0)
                )
                result["layer_global_dpp_identical_heads"] = int(
                    self.prune_diagnostics.get("identical_heads", 0)
                )
                result["layer_global_dpp_mean_overlap_ratio"] = float(
                    self.prune_diagnostics.get("mean_overlap_ratio", 0.0)
                )
                result["layer_global_dpp_mean_score_ratio_vs_greedy"] = float(
                    self.prune_diagnostics.get("mean_score_ratio_vs_greedy", 0.0)
                )
                result["layer_global_dpp_mean_selected_intra_pairwise_sim"] = float(
                    self.prune_diagnostics.get("mean_selected_intra_pairwise_sim", 0.0)
                )
                result["layer_global_dpp_mean_greedy_intra_pairwise_sim"] = float(
                    self.prune_diagnostics.get("mean_greedy_intra_pairwise_sim", 0.0)
                )
                result["layer_global_dpp_mean_selected_cross_pairwise_sim"] = float(
                    self.prune_diagnostics.get("mean_selected_cross_pairwise_sim", 0.0)
                )
                result["layer_global_dpp_mean_greedy_cross_pairwise_sim"] = float(
                    self.prune_diagnostics.get("mean_greedy_cross_pairwise_sim", 0.0)
                )
        return result

    def build_prune_attention_bias(
        self,
        *,
        layer_idx: int,
        seq_len: int,
        q_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor | None:
        if not self.pruned or self.valid is None:
            return None

        full_valid = self._get_valid_positions(layer_idx, seq_len).to(device=device)
        if bool(full_valid.all()):
            return None

        num_heads = full_valid.shape[0]
        bias = torch.zeros((1, num_heads, q_len, seq_len), dtype=dtype, device=device)
        invalid = ~full_valid[:, None, :].expand(num_heads, q_len, seq_len)
        bias.masked_fill_(invalid.unsqueeze(0), torch.finfo(dtype).min)
        return bias

    def _maybe_packed_decode(
        self,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        seq_len: int,
        n_heads_kv_cur: int,
        dim: int,
    ):
        """v3 placeholder — disabled: not compatible with flash_attn_varlen contract.

        Kept to avoid breaking the call site; always returns None so v2 fastpath runs.
        Real packed-decode is implemented in :meth:`decode_attn` via
        flash_attn_with_kvcache.
        """
        return None

    def decode_attn(
        self,
        query_states: torch.Tensor,
        key_states_full: torch.Tensor,
        value_states_full: torch.Tensor,
        layer_idx: int,
        dropout_rate: float,
    ):
        """Single-token decode attention via flash_attn_with_kvcache, treating
        each kv-head as a batch entry. Avoids per-step contiguous packing.

        Returns post-attention tensor shape (1, 1, n_heads_kv * n_group_kv * dim)
        ready for o_proj, or None to indicate fallback.
        """
        if not _GATHER_PACKED_ENABLED:
            return None
        if not self.pruned or self.valid is None:
            return None
        bsz, n_heads_q, q_len, dim = query_states.shape
        if bsz != 1 or q_len != 1:
            return None
        _, n_heads_kv, seq_len, _ = key_states_full.shape
        if n_heads_kv <= 0 or n_heads_q % n_heads_kv != 0:
            return None
        n_group_kv = n_heads_q // n_heads_kv

        device = key_states_full.device
        if device.type != "cuda":
            return None
        fast = self._ensure_gather_fastpath(layer_idx, device, n_heads_kv)
        if fast is None:
            return None
        ctx_end = fast["ctx_end"]
        if ctx_end > seq_len:
            return None
        suf_len = int(seq_len - ctx_end)
        N = fast["N"]
        max_suffix = _GATHER_PACKED_MAX_SUFFIX
        max_len_fixed = fast["max_len_fixed"]
        max_len_total = max_len_fixed + max_suffix

        from gather_profiler import region as _gp_region

        try:
            from flash_attn import flash_attn_with_kvcache
        except ImportError:
            return None

        cache_root = getattr(self, "_gather_packed_cache", None)
        if cache_root is None:
            cache_root = {}
            self._gather_packed_cache = cache_root
        pkey = (layer_idx, device, n_heads_kv, dim, key_states_full.dtype)
        entry = cache_root.get(pkey)

        if entry is None:
            with _gp_region("v3_init"):
                # Allocate per-head padded K/V cache: (N, max_len_total, 1, dim)
                K_cache = torch.zeros((N, max_len_total, 1, dim), dtype=key_states_full.dtype, device=device)
                V_cache = torch.zeros((N, max_len_total, 1, dim), dtype=value_states_full.dtype, device=device)
                # One-time pack ctx rows. Source: key_states_full[0, h, :ctx_end, :] gathered by valid mask.
                # Use precomputed head_id_ctx, ctx_pos_local from fast cache.
                head_id_ctx = fast["head_id_ctx"]
                ctx_pos_local = fast["ctx_pos_local"]
                # Build packed-into-padded destination indices: for the i-th kept entry in
                # head-major order, slot is (head_id_ctx[i], rank_within_head[i]). We need
                # rank within head: for head h, ranks are 0..lens_fixed[h]-1.
                # We have ctx_dst_arange = arange(T_fixed); rank_within_head[i] = arange[i] - cu_lens_fixed[head_id_ctx[i]]
                cu_long = fast["cu_lens_fixed_long"]
                rank_within_head = fast["ctx_dst_arange"] - cu_long[head_id_ctx]
                # Source flat offset into key_states_full[0]: head_id_ctx[i]*seq_len + ctx_pos_local[i]
                K_flat = key_states_full.contiguous().view(-1, dim)
                V_flat = value_states_full.contiguous().view(-1, dim)
                src_flat = head_id_ctx * seq_len + ctx_pos_local
                # Destination flat in (N, max_len_total, 1, dim) viewed as (N*max_len_total, dim):
                dst_flat = head_id_ctx * max_len_total + rank_within_head
                K_cache.view(-1, dim).index_copy_(0, dst_flat, K_flat.index_select(0, src_flat))
                V_cache.view(-1, dim).index_copy_(0, dst_flat, V_flat.index_select(0, src_flat))
                # Keep cache_seqlens in BOTH int32 (for kernel) and int64 (for index ops),
                # sharing storage via two views is not possible across dtypes -- maintain two
                # tensors and update both in step.
                cache_seqlens = fast["lens_fixed"].clone().to(dtype=torch.int32, device=device)  # (N,) int32
                cache_seqlens_long = cache_seqlens.to(torch.long).clone()  # (N,) int64
                # Now consume any suffix tokens (decode tokens) already in key_states_full
                # by appending them sequentially. We treat them as the trailing suf_len rows.
                if suf_len > 0:
                    # Append rows ctx_end..ctx_end+suf_len-1 for each head.
                    for t in range(suf_len):
                        new_K = key_states_full[0, :, ctx_end + t, :].unsqueeze(1).unsqueeze(2)  # (N, 1, 1, D)
                        new_V = value_states_full[0, :, ctx_end + t, :].unsqueeze(1).unsqueeze(2)
                        # Write at position cache_seqlens[h] for each h
                        # Use scatter via index_copy_ on the (N, max_len_total, 1, dim) cache
                        # We avoid the python loop in steady state; this only runs at the
                        # *very first* call with non-zero suf_len (warm-up).
                        idx = cache_seqlens.to(torch.long)  # (N,)
                        for h in range(N):
                            K_cache[h, idx[h], 0].copy_(new_K[h, 0, 0])
                            V_cache[h, idx[h], 0].copy_(new_V[h, 0, 0])
                        cache_seqlens = cache_seqlens + 1
                        cache_seqlens_long = cache_seqlens_long + 1
                entry = {
                    "K_cache": K_cache,
                    "V_cache": V_cache,
                    "cache_seqlens": cache_seqlens,
                    "cache_seqlens_long": cache_seqlens_long,
                    "expected_seq_len": int(seq_len),
                }
                cache_root[pkey] = entry
        else:
            # Steady state: a new token has been added since last call.
            # The new token is at key_states_full[0, :, seq_len-1, :].
            expected = entry["expected_seq_len"]
            if seq_len != expected + 1:
                # Out-of-sync; rebuild from scratch.
                cache_root.pop(pkey, None)
                return self.decode_attn(query_states, key_states_full, value_states_full, layer_idx, dropout_rate)
            with _gp_region("v3_append"):
                # Layer-shared head_base = head_arange * max_len_total (int64), cached once.
                step_cache = getattr(self, "_gather_step_layout_cache", None)
                if step_cache is None:
                    step_cache = {}
                    self._gather_step_layout_cache = step_cache
                hb_key = ("v3_head_base", device, N, max_len_total)
                head_base = step_cache.get(hb_key)
                if head_base is None:
                    head_base = (
                        torch.arange(N, device=device, dtype=torch.long) * max_len_total
                    )
                    step_cache[hb_key] = head_base

                idx_long = entry["cache_seqlens_long"]
                dst_flat = head_base + idx_long
                new_K = key_states_full[0, :, -1, :]  # (N, dim)
                new_V = value_states_full[0, :, -1, :]
                entry["K_cache"].view(-1, dim).index_copy_(0, dst_flat, new_K)
                entry["V_cache"].view(-1, dim).index_copy_(0, dst_flat, new_V)
                entry["cache_seqlens"].add_(1)
                entry["cache_seqlens_long"].add_(1)
                entry["expected_seq_len"] = int(seq_len)

        # Reshape Q: (1, n_heads_q, 1, dim) -> (N, 1, n_group_kv, dim)
        with _gp_region("v3_q_reshape"):
            q = query_states.view(1, n_heads_kv, n_group_kv, 1, dim)  # (1, N, G, 1, D)
            q = q.permute(0, 1, 3, 2, 4).reshape(N, 1, n_group_kv, dim).contiguous()

        with _gp_region("v3_kernel"):
            out = flash_attn_with_kvcache(
                q,
                entry["K_cache"],
                entry["V_cache"],
                cache_seqlens=entry["cache_seqlens"],
                causal=True,
            )
            # out: (N, 1, n_group_kv, dim)

        with _gp_region("v3_output"):
            # Reshape back to (1, 1, n_heads_kv * n_group_kv * dim) order matching baseline:
            # baseline order after v2: view(1, n_heads_kv, 1, n_group_kv, dim).transpose(1,2)
            #                       -> (1, 1, n_heads_kv, n_group_kv, dim).reshape(1,1,-1)
            out = out.view(1, n_heads_kv, n_group_kv, dim).reshape(1, 1, n_heads_kv * n_group_kv * dim)
        return out

    def _ensure_gather_fastpath(self, layer_idx: int, device: torch.device, n_heads_kv_cur: int) -> dict | None:
        """Lazily compute per-layer static structures used by the gather fastpath.

        Cached fields are independent of decode step / current seq_len.
        Returns None if fastpath cannot be used for this layer.
        """
        if not _GATHER_FASTPATH_ENABLED:
            return None
        if self.valid is None:
            return None
        cache_root = getattr(self, "_gather_fastpath_cache", None)
        if cache_root is None:
            cache_root = {}
            self._gather_fastpath_cache = cache_root
        key = (layer_idx, device, n_heads_kv_cur)
        cached = cache_root.get(key)
        if cached is not None:
            return cached

        layer_idx_clamped = min(layer_idx, self.valid.shape[0] - 1)
        ctx_start = int(max(self.ctx_start, 0))
        ctx_end = int(max(self.ctx_end, ctx_start))
        ctx_kept_seq = max(ctx_end - ctx_start, 0)
        valid_layer = self.valid[layer_idx_clamped, :n_heads_kv_cur, :ctx_kept_seq].to(device=device)
        N = valid_layer.shape[0]
        if N == 0:
            return None
        # valid_fixed has shape (N, ctx_end), with prefix=ones(ctx_start) ++ ctx_keep
        if ctx_start > 0:
            prefix_ones = torch.ones((N, ctx_start), dtype=torch.bool, device=device)
            valid_fixed = torch.cat([prefix_ones, valid_layer], dim=-1)
        else:
            valid_fixed = valid_layer
        # Per-head kept positions in [0, ctx_end), packed in head-major order
        nz = valid_fixed.nonzero(as_tuple=False)  # (T_fixed, 2): (head_id, pos)
        head_id_ctx = nz[:, 0].contiguous().to(torch.long)
        ctx_pos_local = nz[:, 1].contiguous().to(torch.long)
        T_fixed = head_id_ctx.numel()
        lens_fixed = valid_fixed.sum(dim=-1, dtype=torch.int32)  # (N,)
        cu_lens_fixed = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=device),
                lens_fixed.cumsum(0, dtype=torch.int32),
            ],
            dim=0,
        ).contiguous()
        max_len_fixed = int(lens_fixed.max().item()) if N > 0 else 0
        ctx_dst_arange = torch.arange(T_fixed, dtype=torch.long, device=device)
        cu_lens_fixed_long = cu_lens_fixed.to(torch.long)
        arange_N1 = torch.arange(N + 1, dtype=torch.int32, device=device)
        head_id_ctx_int32 = head_id_ctx.to(torch.int32)
        cached = {
            "device": device,
            "N": N,
            "ctx_end": ctx_end,
            "T_fixed": T_fixed,
            "head_id_ctx": head_id_ctx,
            "head_id_ctx_int32_seqmul": head_id_ctx_int32,  # unused; kept for clarity
            "ctx_pos_local": ctx_pos_local,
            "lens_fixed": lens_fixed,
            "cu_lens_fixed": cu_lens_fixed,
            "cu_lens_fixed_long": cu_lens_fixed_long,
            "max_len_fixed": max_len_fixed,
            "ctx_dst_arange": ctx_dst_arange,
            "arange_N1": arange_N1,
        }
        cache_root[key] = cached
        return cached

    def prepare(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, int | torch.Tensor]]:
        if not self.pruned or self.valid is None:
            raise ValueError("pruned valid mask is required before prepare()")

        bsz, n_heads_q, q_len, dim = query_states.shape
        _, n_heads_kv, seq_len, _ = key_states.shape
        if n_heads_kv <= 0 or n_heads_q % n_heads_kv != 0:
            raise ValueError(
                f"query heads ({n_heads_q}) must be divisible by kv heads ({n_heads_kv})"
            )

        n_group_kv = max(n_heads_q // n_heads_kv, 1)
        from gather_profiler import region as _gp_region

        # ---- v3: incremental packed buffer (q_len==1 single-token decode) ----
        if (
            _GATHER_PACKED_ENABLED
            and bsz == 1
            and q_len == 1
            and self.valid is not None
        ):
            packed = self._maybe_packed_decode(
                layer_idx,
                key_states,
                value_states,
                seq_len,
                n_heads_kv,
                dim,
            )
            if packed is not None:
                K_packed, V_packed, info_packed = packed
                with _gp_region("p_v3_q_reshape"):
                    q = query_states.contiguous().view(bsz, n_heads_kv, n_group_kv, q_len, dim)
                    q = q.transpose(2, 3).contiguous().view(-1, n_group_kv, dim)
                info_packed["bsz"] = int(bsz)
                info_packed["q_len"] = int(q_len)
                info_packed["head_dim"] = int(dim)
                info_packed["n_group_kv"] = int(n_group_kv)
                info_packed["max_len_q"] = int(q_len)
                return q, K_packed.unsqueeze(1), V_packed.unsqueeze(1), info_packed

        fast = self._ensure_gather_fastpath(layer_idx, key_states.device, n_heads_kv) if bsz == 1 else None
        if fast is not None and fast["ctx_end"] <= seq_len:
            with _gp_region("p_fast_q_reshape"):
                query_states = query_states.contiguous().view(bsz, n_heads_kv, n_group_kv, q_len, dim)
                query_states = query_states.transpose(2, 3).contiguous().view(-1, n_group_kv, dim)

            ctx_end = fast["ctx_end"]
            N = fast["N"]
            suf_len = int(seq_len - ctx_end)
            head_id_ctx = fast["head_id_ctx"]
            ctx_pos_local = fast["ctx_pos_local"]
            T_fixed = fast["T_fixed"]
            cu_lens_fixed = fast["cu_lens_fixed"]
            cu_lens_fixed_long = fast["cu_lens_fixed_long"]
            ctx_dst_arange = fast["ctx_dst_arange"]
            arange_N1 = fast["arange_N1"]
            max_len_fixed = fast["max_len_fixed"]
            device = key_states.device

            with _gp_region("p_fast_indices"):
                step_cache = getattr(self, "_gather_step_layout_cache", None)
                if step_cache is None:
                    step_cache = {}
                    self._gather_step_layout_cache = step_cache
                step_key = (device, N, suf_len)
                step_layout = step_cache.get(step_key)
                if step_layout is None:
                    if suf_len > 0:
                        head_id_suf = (
                            torch.arange(N, device=device, dtype=torch.long).repeat_interleave(suf_len)
                        )
                        suf_off = (
                            torch.arange(suf_len, device=device, dtype=torch.long).repeat(N)
                        )
                        head_id_suf_plus_one = head_id_suf + 1
                        head_id_suf_times_suflen = head_id_suf * suf_len
                        head_id_suf_times_suflen_plus_off = head_id_suf_times_suflen + suf_off
                    else:
                        head_id_suf = None
                        suf_off = None
                        head_id_suf_plus_one = None
                        head_id_suf_times_suflen = None
                        head_id_suf_times_suflen_plus_off = None
                    arange_N1_times_suflen = arange_N1 * int(suf_len)
                    # Cap cache size to prevent unbounded growth.
                    if len(step_cache) > 256:
                        # drop oldest
                        step_cache.pop(next(iter(step_cache)))
                    step_layout = {
                        "head_id_suf": head_id_suf,
                        "suf_off": suf_off,
                        "head_id_suf_plus_one": head_id_suf_plus_one,
                        "head_id_suf_times_suflen": head_id_suf_times_suflen,
                        "head_id_suf_times_suflen_plus_off": head_id_suf_times_suflen_plus_off,
                        "arange_N1_times_suflen": arange_N1_times_suflen,
                    }
                    step_cache[step_key] = step_layout
                head_id_suf = step_layout["head_id_suf"]
                suf_off = step_layout["suf_off"]
                arange_N1_times_suflen = step_layout["arange_N1_times_suflen"]

                # Per-layer per-call (small): scalar-mul on per-layer tensors
                ctx_flat_src = ctx_pos_local + head_id_ctx * seq_len  # (T_fixed,) long
                if suf_len > 0:
                    suf_flat_src = head_id_suf * seq_len + (ctx_end + suf_off)
                    ctx_flat_dst = ctx_dst_arange + head_id_ctx * suf_len
                    suf_flat_dst = (
                        cu_lens_fixed_long[step_layout["head_id_suf_plus_one"]]
                        + step_layout["head_id_suf_times_suflen_plus_off"]
                    )
                    T_total = T_fixed + N * suf_len
                    cu_seqlens_k = cu_lens_fixed + arange_N1_times_suflen
                else:
                    suf_flat_src = None
                    ctx_flat_dst = ctx_dst_arange
                    suf_flat_dst = None
                    T_total = T_fixed
                    cu_seqlens_k = cu_lens_fixed

            with _gp_region("p_fast_kv_pack"):
                K_view = key_states.contiguous().view(-1, dim)
                V_view = value_states.contiguous().view(-1, dim)
                K_packed = K_view.new_empty(T_total, dim)
                V_packed = V_view.new_empty(T_total, dim)
                K_packed.index_copy_(0, ctx_flat_dst, K_view.index_select(0, ctx_flat_src))
                V_packed.index_copy_(0, ctx_flat_dst, V_view.index_select(0, ctx_flat_src))
                if suf_len > 0:
                    K_packed.index_copy_(0, suf_flat_dst, K_view.index_select(0, suf_flat_src))
                    V_packed.index_copy_(0, suf_flat_dst, V_view.index_select(0, suf_flat_src))
                key_states_out = K_packed.unsqueeze(1)
                value_states_out = V_packed.unsqueeze(1)

            cu_seqlens_q = (
                q_len * arange_N1
            ).contiguous() if q_len != 1 else (arange_N1)

            info = {
                "bsz": int(bsz),
                "q_len": int(q_len),
                "head_dim": int(dim),
                "n_heads_kv": int(N),
                "n_group_kv": int(n_group_kv),
                "cu_len_q": cu_seqlens_q.contiguous(),
                "cu_len_k": cu_seqlens_k.contiguous(),
                "max_len_q": int(q_len),
                "max_len_k": int(max_len_fixed + suf_len),
            }
            return query_states, key_states_out, value_states_out, info

        with _gp_region("p1_valid_fetch"):
            valid = self._get_valid_kv(layer_idx, seq_len).to(device=key_states.device)
            valid = valid[:n_heads_kv]
            valid = valid.unsqueeze(0).expand(bsz, -1, -1).contiguous()

        with _gp_region("p2_q_reshape"):
            query_states = query_states.contiguous().view(bsz, n_heads_kv, n_group_kv, q_len, dim)
            query_states = query_states.transpose(2, 3).contiguous().view(-1, n_group_kv, dim)

        with _gp_region("p3_kv_gather"):
            key_states = key_states.contiguous().view(-1, dim)[valid.view(-1)].unsqueeze(1)
            value_states = value_states.contiguous().view(-1, dim)[valid.view(-1)].unsqueeze(1)

        with _gp_region("p4_cu_seqlens"):
            lens_k_head = valid.sum(-1, dtype=torch.int32).reshape(-1).contiguous()
            cu_seqlens_k = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int32, device=valid.device),
                    lens_k_head.cumsum(0, dtype=torch.int32),
                ],
                dim=0,
            ).contiguous()
            cu_seqlens_q = (
                q_len
                * torch.arange(
                bsz * n_heads_kv + 1,
                dtype=torch.int32,
                device=query_states.device,
            )
            ).contiguous()
            max_len_k = int(lens_k_head.max().item())

        info = {
            "bsz": int(bsz),
            "q_len": int(q_len),
            "head_dim": int(dim),
            "n_heads_kv": int(n_heads_kv),
            "n_group_kv": int(n_group_kv),
            "cu_len_q": cu_seqlens_q,
            "cu_len_k": cu_seqlens_k,
            "max_len_q": int(q_len),
            "max_len_k": max_len_k,
        }
        return query_states, key_states, value_states, info

    def _stack_scores(self) -> torch.Tensor:
        device = self._infer_device()
        stacked: list[torch.Tensor] = []
        for layer_idx in range(max(self.n_layers, len(self.score))):
            layer_score = self.score[layer_idx] if layer_idx < len(self.score) else None
            if layer_score is None:
                stacked.append(
                    torch.zeros((self.n_heads_kv, self.ctx_len), dtype=torch.float32, device=device)
                )
            else:
                stacked.append(layer_score.detach().float().squeeze(0))
        return torch.stack(stacked, dim=0)

    def _threshold_global(self, scores: torch.Tensor, ratio: float) -> tuple[torch.Tensor, float]:
        if ratio >= 1:
            return torch.ones_like(scores, dtype=torch.bool), 0.0

        flat = scores.reshape(-1)
        k = max(int(flat.numel() * ratio), 1)
        topk_values = torch.topk(flat, k).values
        threshold = float(topk_values[-1].item())
        valid = scores >= threshold
        return valid, threshold

    def _threshold_uniform(self, scores: torch.Tensor, ratio: float) -> tuple[torch.Tensor, float]:
        select_mode = self._resolve_uniform_head_select_mode()
        if select_mode == "local_head_continuous":
            return self._threshold_uniform_headwise_continuous(scores, ratio)
        if select_mode == "local_head_dpp":
            return self._threshold_uniform_headwise_dpp(scores, ratio)
        if select_mode == "layer_global_dpp_quota":
            return self._threshold_uniform_layer_global_dpp_quota(scores, ratio)

        if ratio >= 1:
            return torch.ones_like(scores, dtype=torch.bool), 0.0

        k = max(int(scores.shape[-1] * ratio), 1)
        valid = torch.zeros_like(scores, dtype=torch.bool)
        topk_indices = torch.topk(scores, k, dim=-1).indices
        valid.scatter_(dim=-1, index=topk_indices, value=True)
        return valid, 0.0

    def _threshold_uniform_layer_linear(
        self,
        scores: torch.Tensor,
        ratio: float,
    ) -> tuple[torch.Tensor, float]:
        if ratio >= 1:
            return torch.ones_like(scores, dtype=torch.bool), 0.0

        num_layers = scores.shape[0]
        num_tokens = scores.shape[-1]
        if num_layers <= 1:
            return self._threshold_uniform(scores, ratio)

        gain = float(max(self.layer_early_ratio_gain_for_prune, 1e-6))
        ratio_first = ratio * gain
        ratio_last = (2.0 * ratio) - ratio_first

        layer_ratios = torch.linspace(ratio_first, ratio_last, steps=num_layers, device=scores.device)
        layer_ratios = layer_ratios.clamp_(min=0.0, max=1.0)

        valid = torch.zeros_like(scores, dtype=torch.bool)
        for layer_idx in range(num_layers):
            k = max(int(math.ceil(num_tokens * float(layer_ratios[layer_idx].item()))), 1)
            k = min(k, num_tokens)
            topk_indices = torch.topk(scores[layer_idx], k, dim=-1).indices
            valid[layer_idx].scatter_(dim=-1, index=topk_indices, value=True)

        return valid, 0.0

    def _threshold_uniform_per_frame(self, scores: torch.Tensor, ratio: float) -> tuple[torch.Tensor, float]:
        if ratio >= 1:
            return torch.ones_like(scores, dtype=torch.bool), 0.0

        frame_count = self.frame_count_for_prune
        if frame_count is None or frame_count <= 0:
            return self._threshold_uniform(scores, ratio)

        ctx_len = scores.shape[-1]
        frame_count = min(frame_count, ctx_len)
        if frame_count <= 1:
            return self._threshold_uniform(scores, ratio)

        # Fast path: equal-size frame segments allow fully vectorized top-k.
        if ctx_len % frame_count == 0:
            seg_len = ctx_len // frame_count
            k = max(int(seg_len * ratio), 1)
            segmented_scores = scores.reshape(*scores.shape[:-1], frame_count, seg_len)
            topk_indices = torch.topk(segmented_scores, k, dim=-1).indices
            segmented_valid = torch.zeros_like(segmented_scores, dtype=torch.bool)
            segmented_valid.scatter_(dim=-1, index=topk_indices, value=True)
            return segmented_valid.reshape_as(scores), 0.0

        base = ctx_len // frame_count
        rem = ctx_len % frame_count

        valid = torch.zeros_like(scores, dtype=torch.bool)
        start = 0
        for frame_idx in range(frame_count):
            seg_len = base + (1 if frame_idx < rem else 0)
            if seg_len <= 0:
                continue

            end = start + seg_len
            segment_scores = scores[..., start:end]
            k = max(int(seg_len * ratio), 1)
            topk_indices = torch.topk(segment_scores, k, dim=-1).indices
            segment_valid = torch.zeros_like(segment_scores, dtype=torch.bool)
            segment_valid.scatter_(dim=-1, index=topk_indices, value=True)
            valid[..., start:end] = segment_valid
            start = end

        return valid, 0.0

    def _threshold_uniform_headwise_continuous(
        self,
        scores: torch.Tensor,
        ratio: float,
    ) -> tuple[torch.Tensor, float]:
        if ratio >= 1:
            return torch.ones_like(scores, dtype=torch.bool), 0.0

        num_layers, num_heads, num_tokens = scores.shape
        if num_heads <= 0 or num_tokens <= 0:
            return torch.zeros_like(scores, dtype=torch.bool), 0.0

        candidate_ratio = float(max(self.head_div_candidate_ratio_for_prune, 1.0))
        diversity_lambda = float(max(self.head_div_lambda_for_prune, 0.0))
        diversity_eps = float(max(self.head_div_eps_for_prune, 1e-12))
        opt_steps = max(int(self.head_continuous_opt_steps_for_prune), 0)
        step_size = float(max(self.head_continuous_step_size_for_prune, 1e-8))
        init_temp = float(max(self.head_continuous_init_temp_for_prune, 1e-6))
        final_temp = float(max(self.head_continuous_final_temp_for_prune, 1e-6))

        valid = torch.zeros_like(scores, dtype=torch.bool)
        keep_k = min(max(int(num_tokens * ratio), 1), num_tokens)
        diagnostics = self._init_continuous_prune_diagnostics()

        for layer_idx in range(num_layers):
            layer_scores = scores[layer_idx]
            if keep_k <= 0:
                continue
            if keep_k >= num_tokens:
                valid[layer_idx] = True
                continue

            candidate_k = min(num_tokens, max(keep_k, int(math.ceil(keep_k * candidate_ratio))))
            candidate_indices = torch.topk(layer_scores, candidate_k, dim=-1).indices
            if keep_k >= candidate_k:
                raise AssertionError(
                    "continuous selection downgraded to greedy: candidate_k_le_keep_k "
                    f"(layer={layer_idx}, keep_k={keep_k}, candidate_k={candidate_k})"
                )
            else:
                candidate_scores = torch.gather(layer_scores, dim=-1, index=candidate_indices).float()
                candidate_keys, gather_reason = self._gather_layer_candidate_keys(
                    layer_idx=layer_idx,
                    candidate_indices=candidate_indices,
                    expected_ctx_len=num_tokens,
                )
                selected_offsets, layer_diag = self._select_continuous_candidate_offsets_batched(
                    candidate_scores=candidate_scores,
                    candidate_keys=candidate_keys,
                    keep_k=keep_k,
                    diversity_lambda=diversity_lambda,
                    eps=diversity_eps,
                    opt_steps=opt_steps,
                    step_size=step_size,
                    init_temp=init_temp,
                    final_temp=final_temp,
                    fallback_reason=gather_reason,
                )
                self._accumulate_continuous_prune_diagnostics(diagnostics, layer_diag)
                selected_indices = torch.gather(
                    candidate_indices,
                    dim=-1,
                    index=selected_offsets.to(device=candidate_indices.device),
                )

            valid[layer_idx].scatter_(dim=-1, index=selected_indices, value=True)

        self.prune_diagnostics = diagnostics
        return valid, 0.0

    def _threshold_uniform_headwise_dpp(
        self,
        scores: torch.Tensor,
        ratio: float,
    ) -> tuple[torch.Tensor, float]:
        if ratio >= 1:
            return torch.ones_like(scores, dtype=torch.bool), 0.0

        num_layers, num_heads, num_tokens = scores.shape
        if num_heads <= 0 or num_tokens <= 0:
            return torch.zeros_like(scores, dtype=torch.bool), 0.0

        candidate_ratio = float(max(self.head_div_candidate_ratio_for_prune, 1.0))
        dpp_eps = float(max(self.head_div_eps_for_prune, 1e-12))
        score_alpha = float(self.head_dpp_score_alpha_for_prune)
        diag_jitter = float(max(self.head_dpp_diag_jitter_for_prune, 1e-12))

        valid = torch.zeros_like(scores, dtype=torch.bool)
        keep_k = min(max(int(num_tokens * ratio), 1), num_tokens)
        diagnostics = (
            self._init_dpp_prune_diagnostics()
            if self.head_dpp_enable_diagnostics_for_prune
            else None
        )

        for layer_idx in range(num_layers):
            layer_scores = scores[layer_idx]
            if keep_k <= 0:
                continue
            if keep_k >= num_tokens:
                valid[layer_idx] = True
                continue

            candidate_k = min(num_tokens, max(keep_k, int(math.ceil(keep_k * candidate_ratio))))
            candidate_indices = torch.topk(layer_scores, candidate_k, dim=-1).indices
            if keep_k >= candidate_k:
                raise AssertionError(
                    "dpp selection requires candidate_k > keep_k "
                    f"(layer={layer_idx}, keep_k={keep_k}, candidate_k={candidate_k})"
                )

            candidate_scores = torch.gather(layer_scores, dim=-1, index=candidate_indices).float()
            candidate_keys, gather_reason = self._gather_layer_candidate_keys(
                layer_idx=layer_idx,
                candidate_indices=candidate_indices,
                expected_ctx_len=num_tokens,
            )
            selected_offsets, layer_diag = self._select_dpp_candidate_offsets_batched(
                candidate_scores=candidate_scores,
                candidate_keys=candidate_keys,
                keep_k=keep_k,
                eps=dpp_eps,
                score_alpha=score_alpha,
                diag_jitter=diag_jitter,
                enable_diagnostics=self.head_dpp_enable_diagnostics_for_prune,
                fallback_reason=gather_reason,
            )
            if diagnostics is not None:
                self._accumulate_dpp_prune_diagnostics(diagnostics, layer_diag)
            selected_indices = torch.gather(
                candidate_indices,
                dim=-1,
                index=selected_offsets.to(device=candidate_indices.device),
            )
            valid[layer_idx].scatter_(dim=-1, index=selected_indices, value=True)

        self.prune_diagnostics = diagnostics
        return valid, 0.0

    def _threshold_uniform_layer_global_dpp_quota(
        self,
        scores: torch.Tensor,
        ratio: float,
    ) -> tuple[torch.Tensor, float]:
        if ratio >= 1:
            return torch.ones_like(scores, dtype=torch.bool), 0.0

        num_layers, num_heads, num_tokens = scores.shape
        if num_heads <= 0 or num_tokens <= 0:
            return torch.zeros_like(scores, dtype=torch.bool), 0.0

        candidate_ratio = float(max(self.head_div_candidate_ratio_for_prune, 1.0))
        dpp_eps = float(max(self.head_div_eps_for_prune, 1e-12))
        score_alpha = float(self.head_dpp_score_alpha_for_prune)
        diag_jitter = float(max(self.head_dpp_diag_jitter_for_prune, 1e-12))

        valid = torch.zeros_like(scores, dtype=torch.bool)
        keep_k = min(max(int(num_tokens * ratio), 1), num_tokens)
        diagnostics = (
            self._init_layer_global_dpp_prune_diagnostics()
            if self.head_dpp_enable_diagnostics_for_prune
            else None
        )

        for layer_idx in range(num_layers):
            layer_scores = scores[layer_idx]
            if keep_k <= 0:
                continue
            if keep_k >= num_tokens:
                valid[layer_idx] = True
                continue

            candidate_k = min(num_tokens, max(keep_k, int(math.ceil(keep_k * candidate_ratio))))
            candidate_indices = torch.topk(layer_scores, candidate_k, dim=-1).indices
            if keep_k >= candidate_k:
                raise AssertionError(
                    "layer-global dpp selection requires candidate_k > keep_k "
                    f"(layer={layer_idx}, keep_k={keep_k}, candidate_k={candidate_k})"
                )

            candidate_scores = torch.gather(layer_scores, dim=-1, index=candidate_indices).float()
            candidate_keys, gather_reason = self._gather_layer_candidate_keys(
                layer_idx=layer_idx,
                candidate_indices=candidate_indices,
                expected_ctx_len=num_tokens,
            )
            selected_offsets, layer_diag = self._select_layer_global_dpp_quota_offsets(
                candidate_scores=candidate_scores,
                candidate_keys=candidate_keys,
                keep_k=keep_k,
                eps=dpp_eps,
                score_alpha=score_alpha,
                diag_jitter=diag_jitter,
                enable_diagnostics=self.head_dpp_enable_diagnostics_for_prune,
                fallback_reason=gather_reason,
            )
            if diagnostics is not None:
                self._accumulate_layer_global_dpp_prune_diagnostics(diagnostics, layer_diag)
            selected_indices = torch.gather(
                candidate_indices,
                dim=-1,
                index=selected_offsets.to(device=candidate_indices.device),
            )
            valid[layer_idx].scatter_(dim=-1, index=selected_indices, value=True)

        self.prune_diagnostics = diagnostics
        return valid, 0.0

    @staticmethod
    def _select_continuous_candidate_offsets_batched(
        *,
        candidate_scores: torch.Tensor,
        candidate_keys: torch.Tensor | None,
        keep_k: int,
        diversity_lambda: float,
        eps: float,
        opt_steps: int,
        step_size: float,
        init_temp: float,
        final_temp: float,
        fallback_reason: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        num_heads, num_candidates = candidate_scores.shape
        keep_k = max(min(int(keep_k), num_candidates), 0)
        diagnostics = RetainCacheLite._init_continuous_prune_diagnostics(num_heads=num_heads)
        if keep_k <= 0:
            return torch.empty((num_heads, 0), dtype=torch.int64, device=candidate_scores.device), diagnostics
        if keep_k >= num_candidates:
            raise AssertionError(
                "continuous selection downgraded to greedy: candidate_k_le_keep_k "
                f"(keep_k={keep_k}, num_candidates={num_candidates})"
            )

        if fallback_reason is not None:
            raise AssertionError(
                "continuous selection downgraded to greedy: "
                f"{fallback_reason} (heads={num_heads}, candidates={num_candidates})"
            )
        if opt_steps <= 0:
            raise AssertionError(
                "continuous selection downgraded to greedy: opt_steps_zero "
                f"(heads={num_heads}, candidates={num_candidates})"
            )
        if candidate_keys is None or candidate_keys.ndim != 3:
            raise AssertionError(
                "continuous selection downgraded to greedy: no_keys "
                f"(heads={num_heads}, candidates={num_candidates})"
            )
        if int(candidate_keys.shape[0]) != num_heads or int(candidate_keys.shape[1]) != num_candidates:
            raise AssertionError(
                "continuous selection downgraded to greedy: shape_mismatch "
                f"(candidate_keys={tuple(candidate_keys.shape)}, heads={num_heads}, candidates={num_candidates})"
            )

        keys = torch.nn.functional.normalize(candidate_keys.float(), p=2.0, dim=-1, eps=eps)
        scores = RetainCacheLite._normalize_continuous_scores(
            candidate_scores.float().to(device=keys.device),
            eps=eps,
        )
        sim = torch.matmul(keys, keys.transpose(-1, -2)).clamp_min_(0.0)
        sim.diagonal(dim1=-2, dim2=-1).zero_()

        logits = (scores / max(init_temp, eps)).detach().clone().requires_grad_(True)
        max_steps = max(int(opt_steps), 0)

        for step_idx in range(max_steps):
            progress = float(step_idx + 1) / float(max_steps)
            temp = (1.0 - progress) * init_temp + progress * final_temp
            temp = max(float(temp), eps)

            weights = float(keep_k) * torch.softmax(logits / temp, dim=-1)
            sim_weights = torch.matmul(sim, weights.unsqueeze(-1)).squeeze(-1)
            rel_term = (weights * scores).sum(dim=-1)
            div_term = (weights * sim_weights).sum(dim=-1)

            objective = rel_term - (float(diversity_lambda) * div_term)
            if not bool(torch.isfinite(objective).all()):
                raise AssertionError(
                    "continuous selection produced non-finite objective "
                    f"(heads={num_heads}, candidates={num_candidates})"
                )

            grad = torch.autograd.grad(objective.sum(), logits, create_graph=False, retain_graph=False)[0]
            if grad is None:
                break

            with torch.no_grad():
                grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                logits = (logits + (float(step_size) * grad)).detach().requires_grad_(True)

        final_temp = max(float(final_temp), eps)
        with torch.no_grad():
            final_weights = float(keep_k) * torch.softmax(logits / final_temp, dim=-1)
            invalid_head_mask = ~torch.isfinite(final_weights).all(dim=-1)
            final_weights = torch.nan_to_num(final_weights, nan=0.0, posinf=0.0, neginf=0.0)
            selected = torch.topk(final_weights, keep_k, dim=-1).indices
            if bool(invalid_head_mask.any()):
                invalid_count = int(invalid_head_mask.sum().item())
                raise AssertionError(
                    "continuous selection produced non-finite final weights "
                    f"(invalid_heads={invalid_count}, total_heads={num_heads})"
                )
            diagnostics["optimized_heads"] += int(num_heads)
            return selected, diagnostics

    @staticmethod
    def _select_dpp_candidate_offsets_batched(
        *,
        candidate_scores: torch.Tensor,
        candidate_keys: torch.Tensor | None,
        keep_k: int,
        eps: float,
        score_alpha: float,
        diag_jitter: float,
        enable_diagnostics: bool = True,
        fallback_reason: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        num_heads, num_candidates = candidate_scores.shape
        keep_k = max(min(int(keep_k), num_candidates), 0)
        diagnostics = (
            RetainCacheLite._init_dpp_prune_diagnostics(num_heads=num_heads)
            if enable_diagnostics
            else {}
        )
        if keep_k <= 0:
            return torch.empty((num_heads, 0), dtype=torch.int64, device=candidate_scores.device), diagnostics
        if keep_k >= num_candidates:
            raise AssertionError(
                "dpp selection requires candidate_k > keep_k "
                f"(keep_k={keep_k}, num_candidates={num_candidates})"
            )
        if fallback_reason is not None:
            raise AssertionError(
                f"dpp selection unavailable: {fallback_reason} "
                f"(heads={num_heads}, candidates={num_candidates})"
            )
        if candidate_keys is None or candidate_keys.ndim != 3:
            raise AssertionError(
                "dpp selection unavailable: no_keys "
                f"(heads={num_heads}, candidates={num_candidates})"
            )
        if int(candidate_keys.shape[0]) != num_heads or int(candidate_keys.shape[1]) != num_candidates:
            raise AssertionError(
                "dpp selection unavailable: shape_mismatch "
                f"(candidate_keys={tuple(candidate_keys.shape)}, heads={num_heads}, candidates={num_candidates})"
            )

        keys = torch.nn.functional.normalize(candidate_keys.float(), p=2.0, dim=-1, eps=eps)
        scores = RetainCacheLite._normalize_dpp_scores_minmax(
            candidate_scores.float().to(device=keys.device),
            eps=eps,
        )
        quality = torch.exp(float(score_alpha) * scores).clamp_min(eps)

        sim = torch.matmul(keys, keys.transpose(-1, -2))
        sim = 0.5 * (sim + sim.transpose(-1, -2))

        kernel = quality.unsqueeze(-1) * sim * quality.unsqueeze(-2)
        kernel.diagonal(dim1=-2, dim2=-1).add_(float(diag_jitter))
        if not bool(torch.isfinite(kernel).all()):
            raise AssertionError(
                "dpp selection produced non-finite kernel "
                f"(heads={num_heads}, candidates={num_candidates})"
            )

        cis = torch.zeros((keep_k, num_heads, num_candidates), device=kernel.device, dtype=kernel.dtype)
        di2s = torch.diagonal(kernel, dim1=-2, dim2=-1).clone()
        if not bool((di2s > 0).any()):
            raise AssertionError(
                "dpp selection produced non-positive diagonal "
                f"(heads={num_heads}, candidates={num_candidates})"
            )

        select_idx = torch.empty((keep_k, num_heads), dtype=torch.long, device=kernel.device)
        head_index = torch.arange(num_heads, device=kernel.device)

        for step_idx in range(keep_k):
            choice = torch.argmax(di2s, dim=-1)
            select_idx[step_idx] = choice
            denom = torch.sqrt(di2s[head_index, choice].clamp_min(eps)).unsqueeze(-1)
            residual = kernel[head_index, choice]
            if step_idx > 0:
                correction = torch.einsum(
                    "th,thn->hn",
                    cis[:step_idx, head_index, choice],
                    cis[:step_idx],
                )
                residual = residual - correction
            eis = residual / denom
            cis[step_idx] = eis
            di2s = di2s - eis.square()
            di2s.clamp_min_(0.0)
            di2s[head_index, choice] = -float("inf")

        selected = torch.sort(select_idx.transpose(0, 1), dim=-1).values
        greedy_offsets = torch.arange(keep_k, device=selected.device, dtype=torch.long).unsqueeze(0)
        greedy_offsets = greedy_offsets.expand(num_heads, -1)
        if enable_diagnostics:
            diagnostics.update(
                RetainCacheLite._compute_dpp_selection_diagnostics(
                    candidate_scores=candidate_scores.to(device=selected.device),
                    normalized_keys=keys,
                    selected_offsets=selected,
                    greedy_offsets=greedy_offsets,
                    keep_k=keep_k,
                    eps=eps,
                )
            )
        return selected, diagnostics

    @staticmethod
    def _select_layer_global_dpp_quota_offsets(
        *,
        candidate_scores: torch.Tensor,
        candidate_keys: torch.Tensor | None,
        keep_k: int,
        eps: float,
        score_alpha: float,
        diag_jitter: float,
        enable_diagnostics: bool = True,
        fallback_reason: str | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        num_heads, num_candidates = candidate_scores.shape
        keep_k = max(min(int(keep_k), num_candidates), 0)
        diagnostics = (
            RetainCacheLite._init_layer_global_dpp_prune_diagnostics(num_heads=num_heads)
            if enable_diagnostics
            else {}
        )
        if keep_k <= 0:
            return torch.empty((num_heads, 0), dtype=torch.int64, device=candidate_scores.device), diagnostics
        if keep_k >= num_candidates:
            raise AssertionError(
                "layer-global dpp selection requires candidate_k > keep_k "
                f"(keep_k={keep_k}, num_candidates={num_candidates})"
            )
        if fallback_reason is not None:
            raise AssertionError(
                f"layer-global dpp selection unavailable: {fallback_reason} "
                f"(heads={num_heads}, candidates={num_candidates})"
            )
        if candidate_keys is None or candidate_keys.ndim != 3:
            raise AssertionError(
                "layer-global dpp selection unavailable: no_keys "
                f"(heads={num_heads}, candidates={num_candidates})"
            )
        if int(candidate_keys.shape[0]) != num_heads or int(candidate_keys.shape[1]) != num_candidates:
            raise AssertionError(
                "layer-global dpp selection unavailable: shape_mismatch "
                f"(candidate_keys={tuple(candidate_keys.shape)}, heads={num_heads}, candidates={num_candidates})"
            )

        keys = torch.nn.functional.normalize(candidate_keys.float(), p=2.0, dim=-1, eps=eps)
        score_norm = RetainCacheLite._normalize_dpp_scores_minmax(
            candidate_scores.float().to(device=keys.device),
            eps=eps,
        )
        quality = torch.exp(float(score_alpha) * score_norm).clamp_min(eps)

        flat_keys = keys.reshape(num_heads * num_candidates, keys.shape[-1])
        flat_quality = quality.reshape(num_heads * num_candidates)
        flat_head_ids = torch.arange(num_heads, device=keys.device, dtype=torch.long)
        flat_head_ids = flat_head_ids.repeat_interleave(num_candidates)

        sim = torch.matmul(flat_keys, flat_keys.transpose(-1, -2))
        sim = 0.5 * (sim + sim.transpose(-1, -2))

        kernel = flat_quality.unsqueeze(-1) * sim * flat_quality.unsqueeze(-2)
        kernel.diagonal().add_(float(diag_jitter))
        if not bool(torch.isfinite(kernel).all()):
            raise AssertionError(
                "layer-global dpp selection produced non-finite kernel "
                f"(heads={num_heads}, candidates={num_candidates})"
            )

        total_keep = num_heads * keep_k
        cis = torch.zeros((total_keep, kernel.shape[0]), device=kernel.device, dtype=kernel.dtype)
        di2s = torch.diagonal(kernel).clone()
        if not bool((di2s > 0).any()):
            raise AssertionError(
                "layer-global dpp selection produced non-positive diagonal "
                f"(heads={num_heads}, candidates={num_candidates})"
            )

        selected = torch.empty((total_keep,), dtype=torch.long, device=kernel.device)
        selected_mask = torch.zeros((kernel.shape[0],), dtype=torch.bool, device=kernel.device)
        head_counts = torch.zeros((num_heads,), dtype=torch.long, device=kernel.device)

        for step_idx in range(total_keep):
            active_mask = (~selected_mask) & (head_counts[flat_head_ids] < keep_k)
            masked_di2s = di2s.masked_fill(~active_mask, -float("inf"))
            choice = torch.argmax(masked_di2s)
            if not bool(torch.isfinite(masked_di2s[choice])):
                raise AssertionError(
                    "layer-global dpp selection exhausted active candidates before filling quota "
                    f"(step={step_idx}, total_keep={total_keep})"
                )

            selected[step_idx] = choice
            selected_mask[choice] = True
            head_counts[flat_head_ids[choice]] += 1

            denom = torch.sqrt(di2s[choice].clamp_min(eps))
            residual = kernel[choice]
            if step_idx > 0:
                residual = residual - (cis[:step_idx, choice].unsqueeze(-1) * cis[:step_idx]).sum(dim=0)
            eis = residual / denom
            cis[step_idx] = eis
            di2s = di2s - eis.square()
            di2s.clamp_min_(0.0)
            di2s[choice] = -float("inf")

        if not bool(torch.equal(head_counts, torch.full_like(head_counts, keep_k))):
            raise AssertionError(
                "layer-global dpp selection failed to satisfy per-head quota "
                f"(head_counts={head_counts.tolist()}, keep_k={keep_k})"
            )

        selected_offsets = torch.empty((num_heads, keep_k), dtype=torch.long, device=kernel.device)
        flat_offsets = selected.remainder(num_candidates)
        selected_heads = flat_head_ids[selected]
        for head_idx in range(num_heads):
            head_selected = flat_offsets[selected_heads == head_idx]
            selected_offsets[head_idx] = torch.sort(head_selected).values
        greedy_offsets = torch.arange(keep_k, device=kernel.device, dtype=torch.long).unsqueeze(0)
        greedy_offsets = greedy_offsets.expand(num_heads, -1)
        selected_sorted = torch.sort(selected_offsets, dim=-1).values
        greedy_sorted = torch.sort(greedy_offsets, dim=-1).values
        overlap_counts = (selected_sorted.unsqueeze(-1) == greedy_sorted.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
        identical_mask = torch.eq(selected_sorted, greedy_sorted).all(dim=-1)
        selected_scores = torch.gather(candidate_scores.to(device=kernel.device), dim=-1, index=selected_offsets)
        greedy_scores = torch.gather(candidate_scores.to(device=kernel.device), dim=-1, index=greedy_offsets)
        score_ratio = selected_scores.sum(dim=-1) / greedy_scores.sum(dim=-1).clamp_min(eps)
        if enable_diagnostics:
            diagnostics.update(
                RetainCacheLite._compute_layer_global_dpp_selection_diagnostics(
                    candidate_scores=candidate_scores.to(device=kernel.device),
                    normalized_keys=keys,
                    selected_offsets=selected_offsets,
                    greedy_offsets=greedy_offsets,
                    keep_k=keep_k,
                    eps=eps,
                )
            )
        return selected_offsets, diagnostics

    @staticmethod
    def _normalize_continuous_scores(candidate_scores: torch.Tensor, eps: float) -> torch.Tensor:
        score_mean = candidate_scores.mean(dim=-1, keepdim=True)
        score_std = candidate_scores.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return (candidate_scores - score_mean) / score_std

    @staticmethod
    def _normalize_dpp_scores_minmax(candidate_scores: torch.Tensor, eps: float) -> torch.Tensor:
        score_min = candidate_scores.amin(dim=-1, keepdim=True)
        score_max = candidate_scores.amax(dim=-1, keepdim=True)
        return (candidate_scores - score_min) / (score_max - score_min).clamp_min(eps)

    def _gather_layer_candidate_keys(
        self,
        *,
        layer_idx: int,
        candidate_indices: torch.Tensor,
        expected_ctx_len: int,
    ) -> tuple[torch.Tensor | None, str | None]:
        layer_keys = self._get_layer_key_states(layer_idx)
        if layer_keys is None:
            return None, "no_keys"
        if layer_keys.ndim != 3:
            return None, "shape_mismatch"
        if candidate_indices.shape[0] > layer_keys.shape[0]:
            return None, "shape_mismatch"

        ctx_start = min(max(self.ctx_start, 0), layer_keys.shape[1])
        ctx_end = min(max(self.ctx_end, ctx_start), layer_keys.shape[1])
        head_ctx_keys = layer_keys[: candidate_indices.shape[0], ctx_start:ctx_end, :]
        if int(head_ctx_keys.shape[1]) != int(expected_ctx_len):
            return None, "shape_mismatch"

        gather_index = candidate_indices.to(device=head_ctx_keys.device).unsqueeze(-1)
        gather_index = gather_index.expand(-1, -1, head_ctx_keys.shape[-1])
        return head_ctx_keys.gather(dim=1, index=gather_index), None

    @staticmethod
    def _init_continuous_prune_diagnostics(*, num_heads: int = 0) -> dict[str, Any]:
        return {
            "selector": "continuous",
            "total_heads": int(num_heads),
            "optimized_heads": 0,
            "fallback_heads": 0,
            "fallback_counts": {
                "candidate_k_le_keep_k": 0,
                "no_keys": 0,
                "shape_mismatch": 0,
                "opt_steps_zero": 0,
                "nan_or_invalid": 0,
            },
        }

    @staticmethod
    def _accumulate_continuous_prune_diagnostics(
        target: dict[str, Any],
        source: dict[str, Any],
    ) -> None:
        target["total_heads"] += int(source.get("total_heads", 0))
        target["optimized_heads"] += int(source.get("optimized_heads", 0))
        target["fallback_heads"] += int(source.get("fallback_heads", 0))
        target_counts = target.setdefault("fallback_counts", {})
        for reason, count in source.get("fallback_counts", {}).items():
            target_counts[reason] = int(target_counts.get(reason, 0)) + int(count)

    @staticmethod
    def _init_dpp_prune_diagnostics(*, num_heads: int = 0) -> dict[str, Any]:
        return {
            "selector": "dpp",
            "total_heads": int(num_heads),
            "changed_heads": 0,
            "identical_heads": 0,
            "overlap_ratio_sum": 0.0,
            "selected_pairwise_sim_sum": 0.0,
            "greedy_pairwise_sim_sum": 0.0,
            "score_ratio_sum": 0.0,
            "mean_overlap_ratio": 0.0,
            "mean_selected_pairwise_sim": 0.0,
            "mean_greedy_pairwise_sim": 0.0,
            "mean_score_ratio_vs_greedy": 0.0,
        }

    @staticmethod
    def _accumulate_dpp_prune_diagnostics(
        target: dict[str, Any],
        source: dict[str, Any],
    ) -> None:
        target["total_heads"] += int(source.get("total_heads", 0))
        target["changed_heads"] += int(source.get("changed_heads", 0))
        target["identical_heads"] += int(source.get("identical_heads", 0))
        target["overlap_ratio_sum"] += float(source.get("overlap_ratio_sum", 0.0))
        target["selected_pairwise_sim_sum"] += float(source.get("selected_pairwise_sim_sum", 0.0))
        target["greedy_pairwise_sim_sum"] += float(source.get("greedy_pairwise_sim_sum", 0.0))
        target["score_ratio_sum"] += float(source.get("score_ratio_sum", 0.0))

        total_heads = max(int(target.get("total_heads", 0)), 1)
        target["mean_overlap_ratio"] = float(target["overlap_ratio_sum"]) / float(total_heads)
        target["mean_selected_pairwise_sim"] = float(target["selected_pairwise_sim_sum"]) / float(total_heads)
        target["mean_greedy_pairwise_sim"] = float(target["greedy_pairwise_sim_sum"]) / float(total_heads)
        target["mean_score_ratio_vs_greedy"] = float(target["score_ratio_sum"]) / float(total_heads)

    @staticmethod
    def _compute_dpp_selection_diagnostics(
        *,
        candidate_scores: torch.Tensor,
        normalized_keys: torch.Tensor,
        selected_offsets: torch.Tensor,
        greedy_offsets: torch.Tensor,
        keep_k: int,
        eps: float,
    ) -> dict[str, Any]:
        num_heads = int(selected_offsets.shape[0])
        keep_k = max(int(keep_k), 1)

        selected_sorted = torch.sort(selected_offsets, dim=-1).values
        greedy_sorted = torch.sort(greedy_offsets, dim=-1).values
        overlap_counts = (selected_sorted.unsqueeze(-1) == greedy_sorted.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
        identical_mask = torch.eq(selected_sorted, greedy_sorted).all(dim=-1)

        selected_scores = torch.gather(candidate_scores, dim=-1, index=selected_offsets)
        greedy_scores = torch.gather(candidate_scores, dim=-1, index=greedy_offsets)
        score_ratio = selected_scores.sum(dim=-1) / greedy_scores.sum(dim=-1).clamp_min(eps)

        selected_keys = torch.gather(
            normalized_keys,
            dim=1,
            index=selected_offsets.unsqueeze(-1).expand(-1, -1, normalized_keys.shape[-1]),
        )
        greedy_keys = torch.gather(
            normalized_keys,
            dim=1,
            index=greedy_offsets.unsqueeze(-1).expand(-1, -1, normalized_keys.shape[-1]),
        )
        selected_pairwise = RetainCacheLite._mean_offdiag_similarity(selected_keys)
        greedy_pairwise = RetainCacheLite._mean_offdiag_similarity(greedy_keys)

        return {
            "selector": "dpp",
            "total_heads": num_heads,
            "changed_heads": int((~identical_mask).sum().item()),
            "identical_heads": int(identical_mask.sum().item()),
            "overlap_ratio_sum": float((overlap_counts.float() / float(keep_k)).sum().item()),
            "selected_pairwise_sim_sum": float(selected_pairwise.sum().item()),
            "greedy_pairwise_sim_sum": float(greedy_pairwise.sum().item()),
            "score_ratio_sum": float(score_ratio.sum().item()),
        }

    @staticmethod
    def _init_layer_global_dpp_prune_diagnostics(*, num_heads: int = 0) -> dict[str, Any]:
        return {
            "selector": "layer_global_dpp_quota",
            "total_heads": int(num_heads),
            "changed_heads": 0,
            "identical_heads": 0,
            "overlap_ratio_sum": 0.0,
            "score_ratio_sum": 0.0,
            "selected_intra_pairwise_sim_sum": 0.0,
            "greedy_intra_pairwise_sim_sum": 0.0,
            "selected_cross_pairwise_sim_sum": 0.0,
            "greedy_cross_pairwise_sim_sum": 0.0,
            "mean_overlap_ratio": 0.0,
            "mean_score_ratio_vs_greedy": 0.0,
            "mean_selected_intra_pairwise_sim": 0.0,
            "mean_greedy_intra_pairwise_sim": 0.0,
            "mean_selected_cross_pairwise_sim": 0.0,
            "mean_greedy_cross_pairwise_sim": 0.0,
        }

    @staticmethod
    def _accumulate_layer_global_dpp_prune_diagnostics(
        target: dict[str, Any],
        source: dict[str, Any],
    ) -> None:
        target["total_heads"] += int(source.get("total_heads", 0))
        target["changed_heads"] += int(source.get("changed_heads", 0))
        target["identical_heads"] += int(source.get("identical_heads", 0))
        target["overlap_ratio_sum"] += float(source.get("overlap_ratio_sum", 0.0))
        target["score_ratio_sum"] += float(source.get("score_ratio_sum", 0.0))
        target["selected_intra_pairwise_sim_sum"] += float(
            source.get("selected_intra_pairwise_sim_sum", 0.0)
        )
        target["greedy_intra_pairwise_sim_sum"] += float(
            source.get("greedy_intra_pairwise_sim_sum", 0.0)
        )
        target["selected_cross_pairwise_sim_sum"] += float(
            source.get("selected_cross_pairwise_sim_sum", 0.0)
        )
        target["greedy_cross_pairwise_sim_sum"] += float(
            source.get("greedy_cross_pairwise_sim_sum", 0.0)
        )

        total_heads = max(int(target.get("total_heads", 0)), 1)
        target["mean_overlap_ratio"] = float(target["overlap_ratio_sum"]) / float(total_heads)
        target["mean_score_ratio_vs_greedy"] = float(target["score_ratio_sum"]) / float(total_heads)
        target["mean_selected_intra_pairwise_sim"] = float(
            target["selected_intra_pairwise_sim_sum"]
        ) / float(total_heads)
        target["mean_greedy_intra_pairwise_sim"] = float(
            target["greedy_intra_pairwise_sim_sum"]
        ) / float(total_heads)
        target["mean_selected_cross_pairwise_sim"] = float(
            target["selected_cross_pairwise_sim_sum"]
        ) / float(total_heads)
        target["mean_greedy_cross_pairwise_sim"] = float(
            target["greedy_cross_pairwise_sim_sum"]
        ) / float(total_heads)

    @staticmethod
    def _compute_layer_global_dpp_selection_diagnostics(
        *,
        candidate_scores: torch.Tensor,
        normalized_keys: torch.Tensor,
        selected_offsets: torch.Tensor,
        greedy_offsets: torch.Tensor,
        keep_k: int,
        eps: float,
    ) -> dict[str, Any]:
        num_heads = int(selected_offsets.shape[0])
        selected_sorted = torch.sort(selected_offsets, dim=-1).values
        greedy_sorted = torch.sort(greedy_offsets, dim=-1).values
        overlap_counts = (selected_sorted.unsqueeze(-1) == greedy_sorted.unsqueeze(-2)).any(dim=-1).sum(dim=-1)
        identical_mask = torch.eq(selected_sorted, greedy_sorted).all(dim=-1)
        selected_scores = torch.gather(candidate_scores, dim=-1, index=selected_offsets)
        greedy_scores = torch.gather(candidate_scores, dim=-1, index=greedy_offsets)
        score_ratio = selected_scores.sum(dim=-1) / greedy_scores.sum(dim=-1).clamp_min(eps)

        selected_keys = torch.gather(
            normalized_keys,
            dim=1,
            index=selected_offsets.unsqueeze(-1).expand(-1, -1, normalized_keys.shape[-1]),
        )
        greedy_keys = torch.gather(
            normalized_keys,
            dim=1,
            index=greedy_offsets.unsqueeze(-1).expand(-1, -1, normalized_keys.shape[-1]),
        )
        selected_intra, selected_cross = RetainCacheLite._mean_intra_cross_pairwise_similarity(selected_keys)
        greedy_intra, greedy_cross = RetainCacheLite._mean_intra_cross_pairwise_similarity(greedy_keys)

        return {
            "selector": "layer_global_dpp_quota",
            "total_heads": num_heads,
            "changed_heads": int((~identical_mask).sum().item()),
            "identical_heads": int(identical_mask.sum().item()),
            "overlap_ratio_sum": float((overlap_counts.float() / float(keep_k)).sum().item()),
            "score_ratio_sum": float(score_ratio.sum().item()),
            "selected_intra_pairwise_sim_sum": float(selected_intra.sum().item()),
            "greedy_intra_pairwise_sim_sum": float(greedy_intra.sum().item()),
            "selected_cross_pairwise_sim_sum": float(selected_cross.sum().item()),
            "greedy_cross_pairwise_sim_sum": float(greedy_cross.sum().item()),
            "mean_overlap_ratio": float((overlap_counts.float() / float(keep_k)).mean().item()),
            "mean_score_ratio_vs_greedy": float(score_ratio.mean().item()),
            "mean_selected_intra_pairwise_sim": float(selected_intra.mean().item()),
            "mean_greedy_intra_pairwise_sim": float(greedy_intra.mean().item()),
            "mean_selected_cross_pairwise_sim": float(selected_cross.mean().item()),
            "mean_greedy_cross_pairwise_sim": float(greedy_cross.mean().item()),
        }

    @staticmethod
    def _mean_intra_cross_pairwise_similarity(selected_keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_heads, keep_k, _ = selected_keys.shape
        if keep_k <= 1:
            intra = torch.zeros(num_heads, device=selected_keys.device, dtype=selected_keys.dtype)
        else:
            intra_sim = torch.matmul(selected_keys, selected_keys.transpose(-1, -2))
            diag_mask = torch.eye(keep_k, dtype=torch.bool, device=selected_keys.device).unsqueeze(0)
            intra = intra_sim.masked_fill(diag_mask, 0.0).sum(dim=(-1, -2)) / float(keep_k * (keep_k - 1))

        if num_heads <= 1:
            cross = torch.zeros(num_heads, device=selected_keys.device, dtype=selected_keys.dtype)
        else:
            head_mean = selected_keys.mean(dim=1)
            cross_sim = torch.matmul(head_mean, head_mean.transpose(-1, -2))
            diag_mask = torch.eye(num_heads, dtype=torch.bool, device=selected_keys.device)
            cross_mean = cross_sim.masked_fill(diag_mask, 0.0).sum(dim=-1) / float(num_heads - 1)
            cross = cross_mean
        return intra, cross

    @staticmethod
    def _mean_offdiag_similarity(selected_keys: torch.Tensor) -> torch.Tensor:
        keep_k = int(selected_keys.shape[1])
        if keep_k <= 1:
            return torch.zeros(selected_keys.shape[0], device=selected_keys.device, dtype=selected_keys.dtype)
        sim = torch.matmul(selected_keys, selected_keys.transpose(-1, -2))
        diag_mask = torch.eye(keep_k, dtype=torch.bool, device=selected_keys.device).unsqueeze(0)
        offdiag = sim.masked_fill(diag_mask, 0.0)
        denom = float(keep_k * (keep_k - 1))
        return offdiag.sum(dim=(-1, -2)) / denom

    def _threshold_adaptive_per_frame(self, scores: torch.Tensor, ratio: float) -> tuple[torch.Tensor, float]:
        if ratio >= 1:
            return torch.ones_like(scores, dtype=torch.bool), 0.0

        frame_count = self.frame_count_for_prune
        if frame_count is None or frame_count <= 0:
            return self._threshold_uniform(scores, ratio)

        ctx_len = scores.shape[-1]
        frame_count = min(frame_count, ctx_len)
        if frame_count <= 1:
            return self._threshold_uniform(scores, ratio)

        segments = self._build_frame_segments(ctx_len, frame_count)
        valid = torch.zeros_like(scores, dtype=torch.bool)

        alpha = float(min(max(self.adaptive_alpha_for_prune, 0.0), 1.0))
        min_keep = max(int(self.adaptive_min_keep_per_frame), 0)
        eps = float(max(self.adaptive_entropy_eps, 1e-12))
        tau = 1.0

        # Preserve exact behavior while skipping adaptive bookkeeping when it
        # mathematically reduces to uniform-per-frame.
        if alpha >= (1.0 - 1e-8):
            return self._threshold_uniform_per_frame(scores, ratio)

        for layer_idx in range(scores.shape[0]):
            layer_keys = self._get_layer_key_states(layer_idx)
            for kv_head_idx in range(scores.shape[1]):
                head_scores = scores[layer_idx, kv_head_idx]
                head_keys = self._get_head_ctx_keys(layer_keys, kv_head_idx)
                bucket_signals_tensor = self._compute_bucket_signals(
                    head_keys,
                    segments=segments,
                    eps=eps,
                    tau=tau,
                )
                uniform_keeps: list[int] = []

                for seg_idx, (start, end) in enumerate(segments):
                    segment = head_scores[start:end]
                    seg_len = int(segment.numel())
                    if seg_len <= 0:
                        uniform_keeps.append(0)
                        continue

                    uniform_keep = max(int(seg_len * ratio), 1)
                    uniform_keeps.append(min(uniform_keep, seg_len))

                keep_target = int(sum(uniform_keeps))
                alloc_keeps = self._allocate_mixed_frame_keeps(
                    seg_lens=[end - start for start, end in segments],
                    uniform_keeps=uniform_keeps,
                    bucket_signals=bucket_signals_tensor,
                    keep_target=keep_target,
                    alpha=alpha,
                    min_keep=min_keep,
                )

                batched_valid = self._select_equal_segments_valid_mask(
                    head_scores=head_scores,
                    segments=segments,
                    alloc_keeps=alloc_keeps,
                )
                if batched_valid is not None:
                    valid[layer_idx, kv_head_idx] = batched_valid
                    continue

                for (start, end), keep_k in zip(segments, alloc_keeps, strict=True):
                    seg_len = end - start
                    if seg_len <= 0 or keep_k <= 0:
                        continue
                    if keep_k >= seg_len:
                        valid[layer_idx, kv_head_idx, start:end] = True
                        continue

                    segment_scores = head_scores[start:end]
                    topk_indices = self._select_segment_indices(segment_scores, keep_k)
                    segment_valid = torch.zeros_like(segment_scores, dtype=torch.bool)
                    segment_valid.scatter_(dim=-1, index=topk_indices, value=True)
                    valid[layer_idx, kv_head_idx, start:end] = segment_valid

        return valid, 0.0

    def _compute_bucket_signals(
        self,
        head_keys: torch.Tensor | None,
        *,
        segments: list[tuple[int, int]],
        eps: float,
        tau: float,
    ) -> torch.Tensor:
        return self._gram_bucket_signals(head_keys, segments=segments, eps=eps, tau=tau)

    def _select_segment_indices(self, segment_scores: torch.Tensor, keep_k: int) -> torch.Tensor:
        selection_scores = self._compute_selection_scores(segment_scores)
        return torch.topk(selection_scores, keep_k, dim=-1).indices

    def _select_equal_segments_valid_mask(
        self,
        *,
        head_scores: torch.Tensor,
        segments: list[tuple[int, int]],
        alloc_keeps: list[int],
    ) -> torch.Tensor | None:
        if not segments:
            return None

        seg_lens = [end - start for start, end in segments]
        if any(seg_len <= 0 for seg_len in seg_lens):
            return None

        seg_len0 = seg_lens[0]
        if not all(seg_len == seg_len0 for seg_len in seg_lens):
            return None

        frame_count = len(segments)
        segmented_scores = head_scores.reshape(frame_count, seg_len0)
        keep_t = torch.tensor(alloc_keeps, dtype=torch.int64, device=head_scores.device)
        if bool((keep_t <= 0).all()):
            return torch.zeros_like(head_scores, dtype=torch.bool)
        if bool((keep_t >= seg_len0).all()):
            return torch.ones_like(head_scores, dtype=torch.bool)

        selection_scores = self._compute_selection_scores(segmented_scores)
        max_keep = int(keep_t.max().item())
        topk_indices = torch.topk(selection_scores, max_keep, dim=-1).indices
        select_mask = (
            torch.arange(max_keep, device=head_scores.device, dtype=torch.int64).unsqueeze(0)
            < keep_t.unsqueeze(1)
        )
        segmented_valid = torch.zeros_like(segmented_scores, dtype=torch.bool)
        segmented_valid.scatter_(dim=-1, index=topk_indices, src=select_mask)
        return segmented_valid.reshape_as(head_scores)

    def _compute_selection_scores(self, segment_scores: torch.Tensor) -> torch.Tensor:
        mode = self.token_select_mode_for_prune
        if mode == "deterministic":
            return segment_scores
        if mode == "gumbel_topk":
            return self._compute_gumbel_selection_scores(segment_scores)
        raise ValueError(f"Unsupported token_select_mode: {mode}")

    def _compute_gumbel_selection_scores(self, segment_scores: torch.Tensor) -> torch.Tensor:
        temperature = float(max(self.token_select_temperature_for_prune, 1e-6))
        noise_scale = float(max(self.token_select_noise_scale_for_prune, 0.0))
        if noise_scale <= 0.0:
            return segment_scores.float() / temperature

        logits = segment_scores.float() / temperature
        generator = None
        if self.token_select_seed_for_prune is not None:
            generator = torch.Generator(device=segment_scores.device)
            generator.manual_seed(self.token_select_seed_for_prune)

        uniform = torch.rand(
            logits.shape,
            dtype=logits.dtype,
            device=logits.device,
            generator=generator,
        ).clamp_(1e-12, 1.0 - 1e-12)
        gumbel = -torch.log(-torch.log(uniform))
        return logits + (noise_scale * gumbel)

    def _get_layer_key_states(self, layer_idx: int) -> torch.Tensor | None:
        if layer_idx >= len(self):
            return None
        try:
            key_states = self[layer_idx][0]
        except Exception:
            return None

        if not isinstance(key_states, torch.Tensor) or key_states.ndim < 4:
            return None
        if key_states.shape[0] <= 0:
            return None
        return key_states[0].detach()

    def _get_head_ctx_keys(self, layer_keys: torch.Tensor | None, kv_head_idx: int) -> torch.Tensor | None:
        if layer_keys is None:
            return None
        if kv_head_idx >= layer_keys.shape[0]:
            return None

        ctx_start = min(max(self.ctx_start, 0), layer_keys.shape[1])
        ctx_end = min(max(self.ctx_end, ctx_start), layer_keys.shape[1])
        return layer_keys[kv_head_idx, ctx_start:ctx_end, :]

    def _gram_bucket_signals(
        self,
        head_keys: torch.Tensor | None,
        *,
        segments: list[tuple[int, int]],
        eps: float,
        tau: float,
    ) -> torch.Tensor:
        frame_count = len(segments)
        if frame_count <= 0:
            return torch.empty(0, dtype=torch.float32)

        if head_keys is None:
            return torch.ones(frame_count, dtype=torch.float32)

        device = head_keys.device
        seg_lens = [end - start for start, end in segments]
        if any(seg_len <= 0 for seg_len in seg_lens):
            signals = torch.ones(frame_count, dtype=head_keys.dtype, device=device)
            for seg_idx, (start, end) in enumerate(segments):
                signals[seg_idx] = self._gram_bucket_signal_single(head_keys[start:end], eps=eps, tau=tau)
            return signals.float().clamp_min_(0.0)

        seg_len0 = seg_lens[0]
        if all(seg_len == seg_len0 for seg_len in seg_lens):
            keys = head_keys.reshape(frame_count, seg_len0, head_keys.shape[-1])
            keys = torch.nn.functional.normalize(keys, p=2.0, dim=-1, eps=eps)
            gram = torch.matmul(keys, keys.transpose(-1, -2))
            diag = torch.diagonal(gram, dim1=-2, dim2=-1)

            diag_err_sq = torch.square(diag - 1.0).sum(dim=-1)
            gram_sq = torch.square(gram).sum(dim=(-2, -1))
            diag_sq = torch.square(diag).sum(dim=-1)
            offdiag_sq = gram_sq - diag_sq
            denom = float(max(seg_len0 * (seg_len0 - 1), 1))
            redundancy = (diag_err_sq + offdiag_sq) / denom
            signals = torch.exp(-tau * redundancy)
            return signals.float().clamp_min_(0.0)

        signals = torch.ones(frame_count, dtype=head_keys.dtype, device=device)
        for seg_idx, (start, end) in enumerate(segments):
            signals[seg_idx] = self._gram_bucket_signal_single(head_keys[start:end], eps=eps, tau=tau)
        return signals.float().clamp_min_(0.0)

    @staticmethod
    def _gram_bucket_signal_single(
        segment_keys: torch.Tensor | None,
        *,
        eps: float,
        tau: float,
    ) -> torch.Tensor:
        if segment_keys is None:
            return torch.tensor(1.0)

        seg_len = int(segment_keys.shape[0])
        if seg_len <= 1:
            return torch.tensor(1.0, device=segment_keys.device, dtype=segment_keys.dtype)

        x = torch.nn.functional.normalize(segment_keys, p=2.0, dim=-1, eps=eps)
        gram = torch.matmul(x, x.transpose(-1, -2))
        diag = torch.diagonal(gram, dim1=-2, dim2=-1)

        diag_err_sq = torch.square(diag - 1.0).sum()
        gram_sq = torch.square(gram).sum()
        diag_sq = torch.square(diag).sum()
        offdiag_sq = gram_sq - diag_sq
        redundancy = (diag_err_sq + offdiag_sq) / float(max(seg_len * (seg_len - 1), 1))
        return torch.exp(-tau * redundancy)

    @staticmethod
    def _build_frame_segments(ctx_len: int, frame_count: int) -> list[tuple[int, int]]:
        if frame_count <= 0 or ctx_len <= 0:
            return [(0, ctx_len)]
        frame_count = min(frame_count, ctx_len)
        base = ctx_len // frame_count
        rem = ctx_len % frame_count
        segments: list[tuple[int, int]] = []
        start = 0
        for frame_idx in range(frame_count):
            seg_len = base + (1 if frame_idx < rem else 0)
            end = start + max(seg_len, 0)
            segments.append((start, end))
            start = end
        return segments

    @staticmethod
    def _allocate_mixed_frame_keeps(
        *,
        seg_lens: list[int],
        uniform_keeps: list[int],
        bucket_signals: list[float] | torch.Tensor,
        keep_target: int,
        alpha: float,
        min_keep: int,
    ) -> list[int]:
        if not seg_lens:
            return []

        signal_t = torch.as_tensor(bucket_signals, dtype=torch.float32).clamp_min(0.0)
        alloc_device = signal_t.device
        seg_lens_t = torch.tensor(seg_lens, dtype=torch.int64, device=alloc_device)
        uniform_keeps_t = torch.tensor(uniform_keeps, dtype=torch.float32, device=alloc_device)

        if signal_t.numel() != seg_lens_t.numel():
            raise ValueError("bucket_signals length must match seg_lens length")

        total_signal = signal_t.sum()
        if bool(total_signal <= 0):
            adaptive_weights_t = torch.full_like(signal_t, 1.0 / float(signal_t.numel()))
        else:
            adaptive_weights_t = signal_t / total_signal

        adaptive_keeps_t = float(keep_target) * adaptive_weights_t
        mixed_t = alpha * uniform_keeps_t + (1.0 - alpha) * adaptive_keeps_t

        lower_bounds_t = torch.clamp(seg_lens_t, min=0, max=max(min_keep, 0))
        upper_bounds_t = torch.clamp(seg_lens_t, min=0)
        total_lower = int(lower_bounds_t.sum().item())
        total_upper = int(upper_bounds_t.sum().item())
        if total_upper <= 0:
            return [0 for _ in seg_lens]

        target = min(max(int(keep_target), total_lower), total_upper)

        clipped_t = torch.minimum(torch.maximum(mixed_t, lower_bounds_t.float()), upper_bounds_t.float())
        keeps_t = torch.floor(clipped_t).to(dtype=torch.int64)
        current = int(keeps_t.sum().item())

        if current < target:
            remainder_t = clipped_t - keeps_t.float()
            eligible = keeps_t < upper_bounds_t
            if bool(eligible.any()):
                candidate_idx = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
                order = torch.argsort(remainder_t[candidate_idx], descending=True)
                for idx in candidate_idx[order].tolist():
                    if current >= target:
                        break
                    if keeps_t[idx] < upper_bounds_t[idx]:
                        keeps_t[idx] += 1
                        current += 1
        elif current > target:
            penalty_t = keeps_t.float() - clipped_t
            eligible = keeps_t > lower_bounds_t
            if bool(eligible.any()):
                candidate_idx = torch.nonzero(eligible, as_tuple=False).squeeze(-1)
                order = torch.argsort(penalty_t[candidate_idx], descending=True)
                for idx in candidate_idx[order].tolist():
                    if current <= target:
                        break
                    if keeps_t[idx] > lower_bounds_t[idx]:
                        keeps_t[idx] -= 1
                        current -= 1

        keeps = keeps_t.tolist()
        if current < target:
            for idx in range(len(keeps)):
                if current >= target:
                    break
                if keeps[idx] < int(upper_bounds_t[idx].item()):
                    keeps[idx] += 1
                    current += 1
        elif current > target:
            for idx in range(len(keeps)):
                if current <= target:
                    break
                if keeps[idx] > int(lower_bounds_t[idx].item()):
                    keeps[idx] -= 1
                    current -= 1

        return keeps

    def _get_valid_positions(self, layer_idx: int, seq_len: int) -> torch.Tensor:
        device = self._infer_device()
        if self.valid is None:
            return torch.ones((self.n_heads, seq_len), dtype=torch.bool, device=device)

        ctx_keep = self._get_valid_kv(layer_idx, seq_len).to(device=device)
        if self.n_group_kv > 1:
            ctx_keep = ctx_keep.repeat_interleave(self.n_group_kv, dim=0)
        ctx_keep = ctx_keep[: self.n_heads]
        return ctx_keep

    def _get_valid_kv(self, layer_idx: int, seq_len: int) -> torch.Tensor:
        device = self._infer_device()
        if self.valid is None:
            return torch.ones((self.n_heads_kv, seq_len), dtype=torch.bool, device=device)

        layer_idx = min(layer_idx, self.valid.shape[0] - 1)
        ctx_start = min(max(self.ctx_start, 0), seq_len)
        ctx_end = min(max(self.ctx_end, ctx_start), seq_len)
        ctx_keep = self.valid[layer_idx, :, : max(ctx_end - ctx_start, 0)].to(device=device)

        prefix = torch.ones((self.n_heads_kv, ctx_start), dtype=torch.bool, device=device)
        suffix = torch.ones((self.n_heads_kv, seq_len - ctx_end), dtype=torch.bool, device=device)
        return torch.cat([prefix, ctx_keep, suffix], dim=-1)

    def _infer_device(self) -> torch.device:
        for layer_score in self.score:
            if layer_score is not None:
                return layer_score.device
        layers = getattr(self, "layers", None)
        if layers:
            first_layer = layers[0]
            keys = getattr(first_layer, "keys", None)
            if keys is not None:
                return keys.device
        if self.prefill_ids is not None:
            return self.prefill_ids.device
        return torch.device("cpu")

    @staticmethod
    def _resolve_num_hidden_layers(config) -> int:
        if config is None:
            return 0
        text_config = RetainCacheLite._resolve_text_config(config)
        if text_config is not None and hasattr(text_config, "num_hidden_layers"):
            return int(text_config.num_hidden_layers)
        if hasattr(config, "num_hidden_layers"):
            return int(config.num_hidden_layers)
        return 0

    @staticmethod
    def _resolve_num_attention_heads(config) -> int:
        if config is None:
            return 0
        text_config = RetainCacheLite._resolve_text_config(config)
        if text_config is not None and hasattr(text_config, "num_attention_heads"):
            return int(text_config.num_attention_heads)
        if hasattr(config, "num_attention_heads"):
            return int(config.num_attention_heads)
        return 0

    @staticmethod
    def _resolve_num_key_value_heads(config) -> int:
        if config is None:
            return 0
        text_config = RetainCacheLite._resolve_text_config(config)
        if text_config is not None and hasattr(text_config, "num_key_value_heads"):
            return int(text_config.num_key_value_heads)
        if hasattr(config, "num_key_value_heads"):
            return int(config.num_key_value_heads)
        return RetainCacheLite._resolve_num_attention_heads(config)

    @staticmethod
    def _resolve_text_config(config):
        if config is None:
            return None
        get_text_config = getattr(config, "get_text_config", None)
        if callable(get_text_config):
            try:
                return get_text_config(decoder=True)
            except TypeError:
                return get_text_config()
        return getattr(config, "text_config", None)

    def _resolve_num_query_groups(self) -> int:
        if self.n_heads_kv <= 0:
            return 1
        if self.n_heads <= 0:
            return 1
        return max(self.n_heads // self.n_heads_kv, 1)
