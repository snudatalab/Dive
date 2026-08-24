from __future__ import annotations

from typing import Any, Callable

import torch
from transformers.cache_utils import DynamicCache

from retain_cache_lite import RetainCacheLite
from video_prompt_templates import get_video_prompt_parts


CACHE_IMPL_CHOICES = ("dynamic", "retain_lite")
PRUNE_MODE_CHOICES = (
    "global",
    "uniform",
    "uniform_per_frame",
    "adaptive_per_frame",
    "uniform_layer_linear",
)
PRUNE_EXEC_CHOICES = ("mask", "gather")
SCORE_QUERY_SCOPE_CHOICES = ("answer_only", "full_tail", "query_only")
SCORE_TASK_MIX_CHOICES = ("single", "caption_only", "qa_only", "caption_and_qa")
HEAD_SELECT_MODE_CHOICES = (
    "greedy",
    "local_head_continuous",
    "local_head_dpp",
    "layer_global_dpp_quota",
)


def move_inputs_to_device(
    inputs: dict[str, Any],
    device: torch.device | str,
    dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            if dtype is not None and torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def tokenize_query_tail(processor, tail_text: str) -> dict[str, torch.Tensor]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("processor does not expose a tokenizer for query tail tokenization")
    return tokenizer(
        [tail_text],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )


def concat_prefill_and_tail(
    prefill_inputs: dict[str, Any],
    tail_inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if "input_ids" not in prefill_inputs or "input_ids" not in tail_inputs:
        raise ValueError("prefill and tail inputs must include input_ids")

    full_inputs: dict[str, torch.Tensor] = {
        "input_ids": torch.cat([prefill_inputs["input_ids"], tail_inputs["input_ids"]], dim=1)
    }

    if "attention_mask" in prefill_inputs and "attention_mask" in tail_inputs:
        full_inputs["attention_mask"] = torch.cat(
            [prefill_inputs["attention_mask"], tail_inputs["attention_mask"]],
            dim=1,
        )
    elif "attention_mask" in prefill_inputs:
        tail_attention_mask = torch.ones_like(tail_inputs["input_ids"], dtype=prefill_inputs["attention_mask"].dtype)
        full_inputs["attention_mask"] = torch.cat(
            [prefill_inputs["attention_mask"], tail_attention_mask],
            dim=1,
        )
    else:
        full_inputs["attention_mask"] = torch.ones_like(full_inputs["input_ids"])

    for key in ("token_type_ids",):
        if key in prefill_inputs and key in tail_inputs:
            full_inputs[key] = torch.cat([prefill_inputs[key], tail_inputs[key]], dim=1)

    return full_inputs


def cache_layer_lengths(cache: DynamicCache) -> tuple[int, ...]:
    return tuple(cache[layer_idx][0].shape[-2] for layer_idx in range(len(cache)))


def build_cache(model, cache_impl: str) -> DynamicCache:
    if cache_impl == "dynamic":
        return DynamicCache(config=model.config)
    if cache_impl == "retain_lite":
        return RetainCacheLite(config=model.config)
    raise ValueError(f"Unsupported cache_impl: {cache_impl}")


def prefill_once(
    model,
    prefill_inputs: dict[str, Any],
    model_dtype: torch.dtype,
    cache_impl: str = "dynamic",
) -> tuple[DynamicCache, int, tuple[int, ...]]:
    cache = build_cache(model, cache_impl)
    if hasattr(cache, "set_prefill_metadata") and "input_ids" in prefill_inputs:
        cache.set_prefill_metadata(prefill_ids=prefill_inputs["input_ids"])
    prefill_model_inputs = move_inputs_to_device(prefill_inputs, model.device, dtype=model_dtype)

    with torch.inference_mode():
        _ = model(
            **prefill_model_inputs,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

    prefill_seq_len = cache.get_seq_length()
    prefill_layer_lens = cache_layer_lengths(cache)
    if prefill_seq_len <= 0:
        raise AssertionError("prefill cache length must be positive after the prefill forward")
    return cache, prefill_seq_len, prefill_layer_lens


def rollback_cache(
    cache: DynamicCache,
    target_seq_len: int,
    target_layer_lens: tuple[int, ...],
) -> None:
    if hasattr(cache, "slice"):
        cache.slice(target_seq_len)
    else:
        cache.crop(target_seq_len)
    if cache.get_seq_length() != target_seq_len:
        raise AssertionError("cache rollback length mismatch after rollback")
    if cache_layer_lengths(cache) != target_layer_lens:
        raise AssertionError("cache rollback tensor shapes do not match the original prefill state")


def generate_from_prefill_cache(
    model,
    processor,
    prefill_inputs: dict[str, Any],
    cache: DynamicCache,
    query_tail_text: str,
    max_new_tokens: int,
    deterministic_generate_kwargs: dict[str, Any],
    model_dtype: torch.dtype,
    decode_generated_fn: Callable[[Any, dict[str, torch.Tensor], torch.Tensor], str],
    update_cache: bool = False,
) -> str:
    seen_token_prev = cache.get_seq_length()
    pre_generate_layer_lens = cache_layer_lengths(cache)

    tail_inputs = tokenize_query_tail(processor, query_tail_text)
    combined_inputs = concat_prefill_and_tail(prefill_inputs, tail_inputs)
    generate_inputs = move_inputs_to_device(combined_inputs, model.device, dtype=model_dtype)

    with torch.inference_mode():
        generated_ids = model.generate(
            **generate_inputs,
            past_key_values=cache,
            max_new_tokens=max_new_tokens,
            **deterministic_generate_kwargs,
        )

    cache_len_after_generate = cache.get_seq_length()
    if cache_len_after_generate <= seen_token_prev:
        raise AssertionError("query/generation did not extend the cache as expected")

    if not update_cache:
        rollback_cache(cache, seen_token_prev, pre_generate_layer_lens)

    return decode_generated_fn(processor, combined_inputs, generated_ids)


def forward_query_from_prefill_cache(
    model,
    prefill_inputs: dict[str, Any],
    processor,
    tail_text: str,
    cache: DynamicCache,
    model_dtype: torch.dtype,
    update_cache: bool = False,
) -> int:
    seen_token_prev = cache.get_seq_length()
    pre_forward_layer_lens = cache_layer_lengths(cache)
    prepared_inputs, tail_inputs, _ = prepare_forward_inputs_from_prefill_cache(
        model=model,
        prefill_inputs=prefill_inputs,
        processor=processor,
        tail_text=tail_text,
        cache=cache,
        model_dtype=model_dtype,
    )

    with torch.inference_mode():
        _ = model(
            **prepared_inputs,
            return_dict=True,
        )

    cache_len_after_forward = cache.get_seq_length()
    if cache_len_after_forward <= seen_token_prev:
        raise AssertionError("query prefill did not extend the cache as expected")

    if not update_cache:
        rollback_cache(cache, seen_token_prev, pre_forward_layer_lens)

    return int(tail_inputs["input_ids"].shape[-1])


def prepare_forward_inputs_from_prefill_cache(
    model,
    prefill_inputs: dict[str, Any],
    processor,
    tail_text: str,
    cache: DynamicCache,
    model_dtype: torch.dtype,
) -> tuple[dict[str, Any], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    tail_inputs = tokenize_query_tail(processor, tail_text)
    combined_inputs = concat_prefill_and_tail(prefill_inputs, tail_inputs)
    full_inputs = move_inputs_to_device(combined_inputs, model.device, dtype=model_dtype)
    prefill_model_inputs = move_inputs_to_device(prefill_inputs, model.device, dtype=model_dtype)
    cache_position = torch.arange(
        cache.get_seq_length(),
        combined_inputs["input_ids"].shape[1],
        device=model.device,
        dtype=torch.long,
    )
    prepare_kwargs = {
        **prefill_model_inputs,
        **full_inputs,
        "past_key_values": cache,
        "use_cache": True,
        "cache_position": cache_position,
    }
    prepared_inputs = model.prepare_inputs_for_generation(
        **prepare_kwargs,
    )
    return prepared_inputs, tail_inputs, combined_inputs


def clone_score_tensors(
    score_tensors: list[torch.Tensor | None],
    *,
    to_cpu: bool = True,
) -> list[torch.Tensor]:
    cloned: list[torch.Tensor] = []
    for tensor in score_tensors:
        if tensor is None:
            continue
        copied = tensor.detach().clone()
        if to_cpu:
            copied = copied.cpu()
        cloned.append(copied)
    return cloned


def assign_score_tensors_to_cache(
    cache: DynamicCache,
    score_tensors: list[torch.Tensor],
) -> None:
    if not hasattr(cache, "score"):
        raise ValueError("cache implementation does not expose score storage")
    cache.score = [tensor.detach().clone() for tensor in score_tensors]


def resolve_prefill_ctx_bounds(
    processor,
    model_name: str,
    prefill_inputs: dict[str, Any],
) -> tuple[int, int]:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("processor does not expose a tokenizer for ctx boundary resolution")

    prompt_parts = get_video_prompt_parts(model_name)
    prefix_len = tokenizer(
        [prompt_parts.prefix_before_ctx],
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].shape[-1]
    suffix_len = tokenizer(
        [prompt_parts.suffix_after_video_before_query],
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].shape[-1]
    prefill_len = int(prefill_inputs["input_ids"].shape[-1])

    ctx_start = min(max(int(prefix_len), 0), prefill_len)
    ctx_end = max(ctx_start, min(prefill_len - int(suffix_len), prefill_len))
    return ctx_start, ctx_end


def summarize_score_tensors(score_tensors: list[torch.Tensor]) -> str:
    if not score_tensors:
        return "layers=0"
    layer_shapes = [tuple(tensor.shape) for tensor in score_tensors]
    ctx_lens = sorted({shape[-1] for shape in layer_shapes})
    max_scores = [float(tensor.max().item()) for tensor in score_tensors]
    mean_scores = [float(tensor.mean().item()) for tensor in score_tensors]
    return (
        f"layers={len(score_tensors)} "
        f"ctx_lens={ctx_lens} "
        f"layer0_shape={layer_shapes[0]} "
        f"score_max={max(max_scores):.4f} "
        f"score_mean={sum(mean_scores) / len(mean_scores):.4f}"
    )


def summarize_prune_result(prune_result: dict[str, Any]) -> str:
    return (
        f"mode={prune_result['mode']} "
        f"exec={prune_result.get('exec', 'mask')} "
        f"requested_ratio={prune_result['requested_ratio']:.4f} "
        f"actual_ratio={prune_result['actual_ratio']:.4f} "
        f"ctx_len={prune_result['ctx_len']} "
        f"layers={prune_result['layers']} "
        f"threshold={prune_result['threshold']:.6f}"
    )


def summarize_score_debug_report(debug_report: dict[str, Any]) -> str:
    layer_idx = debug_report["layer_idx"]
    recorded_shape = tuple(debug_report["recorded_score"].shape)
    cache_shape = tuple(debug_report["cache_score"].shape)
    return (
        f"layer={layer_idx} "
        f"recorded_shape={recorded_shape} "
        f"cache_shape={cache_shape} "
        f"manual_vs_recorded_max_abs_diff={debug_report['manual_vs_recorded_max_abs_diff']:.6f} "
        f"manual_vs_cache_max_abs_diff={debug_report['manual_vs_cache_max_abs_diff']:.6f}"
    )


def verify_score_debug_record(cache: DynamicCache) -> dict[str, Any]:
    debug_record = getattr(cache, "debug_score_record", None)
    if debug_record is None:
        raise AssertionError("debug score record was not captured")
    if not hasattr(cache, "manual_score_from_debug_record"):
        raise ValueError("cache implementation does not support manual score verification")

    layer_idx = int(debug_record["layer_idx"])
    recorded_score = debug_record["score"]
    manual_score = cache.manual_score_from_debug_record(debug_record).cpu()
    cache_score = cache.score[layer_idx]
    if cache_score is None:
        raise AssertionError("cache score for debug layer is missing")
    cache_score = cache_score.detach().cpu()

    if manual_score.shape != recorded_score.shape:
        raise AssertionError("manual score shape does not match recorded score shape")
    if cache_score.shape != recorded_score.shape:
        raise AssertionError("cache score shape does not match recorded score shape")

    manual_vs_recorded = (manual_score - recorded_score).abs().max().item()
    manual_vs_cache = (manual_score - cache_score).abs().max().item()

    return {
        "layer_idx": layer_idx,
        "recorded_score": recorded_score,
        "manual_score": manual_score,
        "cache_score": cache_score,
        "manual_vs_recorded_max_abs_diff": float(manual_vs_recorded),
        "manual_vs_cache_max_abs_diff": float(manual_vs_cache),
    }


def score_from_prefill_cache(
    model,
    processor,
    prefill_inputs: dict[str, Any],
    cache: DynamicCache,
    task_query_tail_text: str,
    answer_text: str,
    model_dtype: torch.dtype,
    update_cache: bool = False,
    debug_score_layer: int | None = None,
    ctx_bounds: tuple[int, int] | None = None,
    query_scope: str = "answer_only",
    keep_score_on_device: bool = False,
    strip_role_bridge: bool = False,
    role_bridge_text: str | None = None,
) -> tuple[list[torch.Tensor], dict[str, Any] | None]:
    if not hasattr(cache, "init_score"):
        raise ValueError("cache implementation does not support score collection")
    if query_scope not in SCORE_QUERY_SCOPE_CHOICES:
        raise ValueError(f"Unsupported query_scope: {query_scope}")
    if query_scope != "query_only" and (not answer_text or not answer_text.strip()):
        raise ValueError("answer_text must be non-empty for score collection")

    seen_token_prev = cache.get_seq_length()
    pre_forward_layer_lens = cache_layer_lengths(cache)

    task_query_tail_for_score = task_query_tail_text
    if (
        strip_role_bridge
        and query_scope == "full_tail"
        and role_bridge_text
        and task_query_tail_for_score.endswith(role_bridge_text)
    ):
        task_query_tail_for_score = task_query_tail_for_score[: -len(role_bridge_text)]

    if query_scope == "query_only":
        answer_tail_text = task_query_tail_text
    else:
        answer_tail_text = f"{task_query_tail_for_score}{answer_text}"
    prepared_inputs, _, _ = prepare_forward_inputs_from_prefill_cache(
        model=model,
        prefill_inputs=prefill_inputs,
        processor=processor,
        tail_text=answer_tail_text,
        cache=cache,
        model_dtype=model_dtype,
    )
    query_prefix_len = 0
    if query_scope == "answer_only":
        query_prefix_len = tokenize_query_tail(processor, task_query_tail_text)["input_ids"].shape[-1]
    ctx_start, ctx_end = (0, seen_token_prev) if ctx_bounds is None else ctx_bounds
    cache.init_score(
        ctx_start=ctx_start,
        ctx_end=ctx_end,
        query_start=query_prefix_len,
        debug_capture=debug_score_layer is not None,
        debug_layer_idx=0 if debug_score_layer is None else debug_score_layer,
    )

    with torch.inference_mode():
        _ = model(
            **prepared_inputs,
            return_dict=True,
        )

    cache_len_after_forward = cache.get_seq_length()
    if cache_len_after_forward <= seen_token_prev:
        raise AssertionError("score forward did not extend the cache as expected")

    score_tensors = clone_score_tensors(cache.score, to_cpu=not keep_score_on_device)
    if not score_tensors:
        raise AssertionError("score collection did not produce any layer scores")
    expected_ctx_len = max(ctx_end - ctx_start, 0)
    if any(tensor.shape[-1] != expected_ctx_len for tensor in score_tensors):
        raise AssertionError("score tensor context length does not match the prefill length")

    debug_report = None
    if debug_score_layer is not None:
        debug_report = verify_score_debug_record(cache)

    cache.get_score = False
    if not update_cache:
        rollback_cache(cache, seen_token_prev, pre_forward_layer_lens)

    return score_tensors, debug_report


def merge_score_tensors_max_(
    aggregate_scores: list[torch.Tensor],
    aggregate_sources: list[torch.Tensor],
    new_scores: list[torch.Tensor],
    *,
    task_idx: int,
) -> tuple[int, int]:
    updated_pairs = 0
    total_pairs = 0
    for layer_idx, (aggregate_score, aggregate_source, new_score) in enumerate(
        zip(aggregate_scores, aggregate_sources, new_scores, strict=True)
    ):
        if aggregate_score.shape != new_score.shape:
            raise AssertionError(
                f"aggregate score shape mismatch at layer {layer_idx}: "
                f"{tuple(aggregate_score.shape)} != {tuple(new_score.shape)}"
            )
        update_mask = new_score > aggregate_score
        updated_pairs += int(update_mask.sum().item())
        total_pairs += int(update_mask.numel())
        aggregate_score.copy_(torch.maximum(aggregate_score, new_score))
        aggregate_source[update_mask] = int(task_idx)
    return updated_pairs, total_pairs


def summarize_score_source_maps(
    source_maps: list[torch.Tensor],
    task_labels: list[str],
) -> str:
    if not source_maps:
        return "pairs=0"

    counts = [0 for _ in task_labels]
    total = 0
    for source_map in source_maps:
        source_flat = source_map.view(-1)
        bincount = torch.bincount(source_flat, minlength=len(task_labels))
        total += int(source_flat.numel())
        for idx, value in enumerate(bincount.tolist()[: len(task_labels)]):
            counts[idx] += int(value)

    if total <= 0:
        return "pairs=0"
    return " ".join(
        f"{label}:{(count / total) * 100:.1f}%"
        for label, count in zip(task_labels, counts, strict=True)
    )


def build_score_source_stats(
    source_maps: list[torch.Tensor],
    task_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    if not source_maps:
        return {
            "total_pairs": 0,
            "task_counts": {},
            "kind_counts": {},
            "dominant_label": None,
            "dominant_fraction": 0.0,
        }

    task_labels = [str(task_report["label"]) for task_report in task_reports]
    task_kinds = [str(task_report["kind"]) for task_report in task_reports]
    counts = [0 for _ in task_labels]
    total = 0

    for source_map in source_maps:
        source_flat = source_map.view(-1)
        bincount = torch.bincount(source_flat, minlength=len(task_labels))
        total += int(source_flat.numel())
        for idx, value in enumerate(bincount.tolist()[: len(task_labels)]):
            counts[idx] += int(value)

    task_counts = {
        label: count for label, count in zip(task_labels, counts, strict=True)
    }
    kind_counts: dict[str, int] = {}
    for kind, count in zip(task_kinds, counts, strict=True):
        kind_counts[kind] = kind_counts.get(kind, 0) + count

    dominant_label = None
    dominant_fraction = 0.0
    if total > 0 and task_counts:
        dominant_label, dominant_count = max(task_counts.items(), key=lambda item: item[1])
        dominant_fraction = dominant_count / total

    return {
        "total_pairs": total,
        "task_counts": task_counts,
        "kind_counts": kind_counts,
        "dominant_label": dominant_label,
        "dominant_fraction": dominant_fraction,
    }


def summarize_score_source_groups(source_stats: dict[str, Any]) -> str:
    total = int(source_stats.get("total_pairs", 0))
    if total <= 0:
        return "pairs=0"

    kind_counts = source_stats.get("kind_counts", {})
    labels = []
    for kind in ("caption", "qa", "single"):
        count = int(kind_counts.get(kind, 0))
        if count > 0:
            labels.append(f"{kind}:{(count / total) * 100:.1f}%")
    if not labels:
        return "pairs=0"
    return " ".join(labels)


def summarize_score_source_dominant(source_stats: dict[str, Any]) -> str:
    total = int(source_stats.get("total_pairs", 0))
    dominant_label = source_stats.get("dominant_label")
    dominant_fraction = float(source_stats.get("dominant_fraction", 0.0))
    if total <= 0 or dominant_label is None:
        return "pairs=0"
    return f"{dominant_label}:{dominant_fraction * 100:.1f}%"


def score_from_prefill_cache_multi_tasks(
    model,
    processor,
    prefill_inputs: dict[str, Any],
    cache: DynamicCache,
    tasks: list[dict[str, str]],
    model_dtype: torch.dtype,
    ctx_bounds: tuple[int, int] | None = None,
    query_scope: str = "answer_only",
    strip_role_bridge: bool = False,
    role_bridge_text: str | None = None,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    if not tasks:
        raise ValueError("tasks must be non-empty for multi-task score collection")

    aggregate_scores: list[torch.Tensor] | None = None
    aggregate_sources: list[torch.Tensor] | None = None
    task_reports: list[dict[str, Any]] = []
    task_labels: list[str] = []

    for task_idx, task in enumerate(tasks):
        label = str(task.get("label") or f"task{task_idx}")
        task_labels.append(label)
        task_query_tail_text = task["task_query_tail_text"]
        answer_text = task["answer"]
        task_scores, _ = score_from_prefill_cache(
            model=model,
            processor=processor,
            prefill_inputs=prefill_inputs,
            cache=cache,
            task_query_tail_text=task_query_tail_text,
            answer_text=answer_text,
            model_dtype=model_dtype,
            update_cache=False,
            debug_score_layer=None,
            ctx_bounds=ctx_bounds,
            query_scope=query_scope,
            keep_score_on_device=True,
            strip_role_bridge=strip_role_bridge,
            role_bridge_text=role_bridge_text,
        )

        updated_pairs = sum(int(tensor.numel()) for tensor in task_scores)
        total_pairs = updated_pairs
        if aggregate_scores is None:
            aggregate_scores = [tensor.detach().clone() for tensor in task_scores]
            aggregate_sources = [
                torch.full(
                    tensor.shape,
                    fill_value=task_idx,
                    dtype=torch.int32,
                    device=tensor.device,
                )
                for tensor in task_scores
            ]
        else:
            updated_pairs, total_pairs = merge_score_tensors_max_(
                aggregate_scores,
                aggregate_sources,
                task_scores,
                task_idx=task_idx,
            )
        task_reports.append(
            {
                "label": label,
                "kind": task.get("kind", "task"),
                "question": task["question"],
                "answer": answer_text,
                "score_summary": summarize_score_tensors(task_scores),
                "updated_pairs": updated_pairs,
                "updated_fraction": (updated_pairs / total_pairs) if total_pairs > 0 else 0.0,
                "rollback_ok": True,
            }
        )

    if aggregate_scores is None or aggregate_sources is None:
        raise AssertionError("multi-task score aggregation did not produce any scores")

    report = {
        "query_scope": query_scope,
        "task_reports": task_reports,
        "aggregate_summary": summarize_score_tensors(aggregate_scores),
        "source_summary": summarize_score_source_maps(aggregate_sources, task_labels),
    }
    source_stats = build_score_source_stats(aggregate_sources, task_reports)
    report["source_group_summary"] = summarize_score_source_groups(source_stats)
    report["source_dominant_summary"] = summarize_score_source_dominant(source_stats)
    return clone_score_tensors(aggregate_scores), report


def run_prefill_generation(
    model,
    processor,
    prefill_inputs: dict[str, Any],
    query_tail_text: str,
    max_new_tokens: int,
    deterministic_generate_kwargs: dict[str, Any],
    model_dtype: torch.dtype,
    decode_generated_fn: Callable[[Any, dict[str, torch.Tensor], torch.Tensor], str],
    cache_impl: str = "dynamic",
) -> str:
    cache, _, _ = prefill_once(
        model=model,
        prefill_inputs=prefill_inputs,
        model_dtype=model_dtype,
        cache_impl=cache_impl,
    )
    return generate_from_prefill_cache(
        model=model,
        processor=processor,
        prefill_inputs=prefill_inputs,
        cache=cache,
        query_tail_text=query_tail_text,
        max_new_tokens=max_new_tokens,
        deterministic_generate_kwargs=deterministic_generate_kwargs,
        model_dtype=model_dtype,
        decode_generated_fn=decode_generated_fn,
        update_cache=False,
    )
