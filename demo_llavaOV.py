import argparse
import random
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

DEMO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = DEMO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from demo_utils import measure_latency
from prefill_cache_utils import (
    assign_score_tensors_to_cache,
    generate_from_prefill_cache,
    prefill_once,
    resolve_prefill_ctx_bounds,
    score_from_prefill_cache_multi_tasks,
    summarize_prune_result,
    summarize_score_tensors,
)
from score_attention_patch import apply_score_attention_patch
from video_prompt_templates import get_video_prompt_parts

MODEL_ID = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
FRAME_DIR = SRC_ROOT / "img"
MAX_NEW_TOKENS = 96
PRUNE_RATIO = 0.05
PRUNE_MODE = "uniform"
SCORE_QA_PACK_SIZE = 1
SCORE_QA_NUM_PACKS = 4
SCORE_QA_SAMPLE_SEED = 0
SCORE_QUERY_SCOPE = "full_tail"
HEAD_SELECT_MODE = "local_head_dpp"
HEAD_DIV_CANDIDATE_RATIO = 2.0
HEAD_DPP_SCORE_ALPHA = 4.0
HEAD_DPP_DIAG_JITTER = 1e-6
COMPARE_GENERATE_KWARGS = {"do_sample": False}
DEFAULT_FINAL_QUESTION = "Describe the video in temporal order."
PROXY_QA_TASKS = [
    {
        "question": "What happens between the person starts drilling holes into the wooden slats using a drill and the person adjusts the position of the wooden slats on the frame?",
        "answer": "the person continues to drill more holes into the wooden slats",
    },
    {
        "question": "What happens between the person continues to drill more holes into the wooden slats and the person makes final adjustments to the wooden slats on the frame?",
        "answer": "the person adjusts the position of the wooden slats on the frame",
    },
    {
        "question": "What happens between the person adjusts the position of the wooden slats on the frame and the person secures the wooden slats in place with screws?",
        "answer": "the person makes final adjustments to the wooden slats on the frame",
    },
    {
        "question": "What happens between the person makes final adjustments to the wooden slats on the frame and the person completes the assembly of the wooden slats on the frame?",
        "answer": "the person secures the wooden slats in place with screws",
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prune-exec", choices=("mask", "gather"), default="gather")
    parser.add_argument("--final-question", default=DEFAULT_FINAL_QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--print-response", action="store_true")
    return parser.parse_args()


def load_model_and_processor():
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    apply_score_attention_patch(model)
    return model, processor


def is_frame_sequence(video_input) -> bool:
    return isinstance(video_input, Sequence) and not isinstance(video_input, (str, bytes))


def load_frame_paths() -> list[str]:
    frame_paths = sorted(str(path) for path in FRAME_DIR.glob("*.png"))
    if len(frame_paths) != 16:
        raise ValueError(f"Expected exactly 16 frames under {FRAME_DIR}, found {len(frame_paths)}")
    return frame_paths


def build_manual_text(query: str) -> str:
    prompt_parts = get_video_prompt_parts(MODEL_ID)
    return prompt_parts.render_prompt(query)


def build_prefill_text() -> str:
    prompt_parts = get_video_prompt_parts(MODEL_ID)
    return prompt_parts.render_prefill_prompt()


def build_query_tail_text(query: str) -> str:
    prompt_parts = get_video_prompt_parts(MODEL_ID)
    return prompt_parts.render_query_tail(query)


def build_manual_inputs(processor, video_input, query: str):
    manual_text = build_manual_text(query)
    if not is_frame_sequence(video_input):
        raise ValueError("demo expects a pre-extracted frame sequence")
    return processor(
        text=[manual_text],
        videos=[[list(video_input)]],
        do_sample_frames=False,
        return_tensors="pt",
    )


def build_prefill_inputs(processor, video_input):
    prefill_text = build_prefill_text()
    if not is_frame_sequence(video_input):
        raise ValueError("demo expects a pre-extracted frame sequence")
    return processor(
        text=[prefill_text],
        videos=[[list(video_input)]],
        do_sample_frames=False,
        return_tensors="pt",
    )


def decode_generated(processor, inputs, output_ids):
    trimmed = output_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]


def generate_text(model, processor, inputs, max_new_tokens: int):
    model_inputs = inputs.to(model.device, torch.float16)
    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        **COMPARE_GENERATE_KWARGS,
    )
    return decode_generated(processor, model_inputs, output_ids)


def build_selected_score_tasks() -> list[dict[str, str]]:
    prompt_parts = get_video_prompt_parts(MODEL_ID)
    qa_pool = list(PROXY_QA_TASKS)
    total_needed = min(len(qa_pool), SCORE_QA_PACK_SIZE * SCORE_QA_NUM_PACKS)
    sampled_indices = sorted(
        random.Random(SCORE_QA_SAMPLE_SEED).sample(range(len(qa_pool)), k=total_needed)
    )
    sampled_qas = [qa_pool[idx] for idx in sampled_indices]
    packs = [
        sampled_qas[idx : idx + SCORE_QA_PACK_SIZE]
        for idx in range(0, len(sampled_qas), SCORE_QA_PACK_SIZE)
    ]

    selected_tasks = []
    for idx, qa_pack in enumerate(packs[:SCORE_QA_NUM_PACKS], start=1):
        first_qa = qa_pack[0]
        followup_pairs = [(qa["question"], qa["answer"]) for qa in qa_pack[1:]]
        selected_tasks.append(
            {
                "label": f"qa_pack{idx}",
                "kind": "qa",
                "question": first_qa["question"],
                "answer": prompt_parts.render_answer_with_followups(
                    first_qa["answer"],
                    followup_pairs,
                ),
                "task_query_tail_text": build_query_tail_text(first_qa["question"]),
            }
        )
    return selected_tasks


def format_selected_qa_report(selected_tasks: list[dict[str, str]]) -> str:
    lines = [
        f"[Selected Proxy QA] requested={SCORE_QA_NUM_PACKS} available={len(PROXY_QA_TASKS)} selected={len(selected_tasks)}"
    ]
    for task in selected_tasks:
        lines.append(f"  - {task['label']}: {task['question']}")
    return "\n".join(lines)


def format_dpp_summary(prune_result: dict[str, float | int | str]) -> str:
    return (
        "[DPP] "
        f"total_heads={prune_result.get('dpp_total_heads', 0)} "
        f"changed_heads={prune_result.get('dpp_changed_heads', 0)} "
        f"identical_heads={prune_result.get('dpp_identical_heads', 0)} "
        f"mean_overlap_ratio={float(prune_result.get('dpp_mean_overlap_ratio', 0.0)):.4f} "
        f"mean_score_ratio_vs_greedy={float(prune_result.get('dpp_mean_score_ratio_vs_greedy', 0.0)):.4f}"
    )


def run_demo(args):
    model, processor = load_model_and_processor()
    frame_paths = load_frame_paths()
    selected_score_tasks = build_selected_score_tasks()

    (manual_inputs, prep_manual) = measure_latency(
        build_manual_inputs,
        processor,
        frame_paths,
        args.final_question,
    )
    (fresh_output, fresh_gen) = measure_latency(
        generate_text,
        model,
        processor,
        manual_inputs,
        args.max_new_tokens,
    )

    (prefill_inputs, prep_prefill) = measure_latency(
        build_prefill_inputs,
        processor,
        frame_paths,
    )
    ((cache, _, _), prefill_cache) = measure_latency(
        prefill_once,
        model,
        prefill_inputs,
        torch.float16,
        "retain_lite",
    )
    if hasattr(cache, "set_score_aggregation_mode"):
        cache.set_score_aggregation_mode("max")

    ctx_bounds = resolve_prefill_ctx_bounds(processor, MODEL_ID, prefill_inputs)
    ((score_tensors, _), score_forward) = measure_latency(
        score_from_prefill_cache_multi_tasks,
        model,
        processor,
        prefill_inputs,
        cache,
        selected_score_tasks,
        torch.float16,
        ctx_bounds,
        SCORE_QUERY_SCOPE,
    )
    assign_score_tensors_to_cache(cache, score_tensors)

    if hasattr(cache, "set_frame_layout_for_prune"):
        cache.set_frame_layout_for_prune(frame_count=len(frame_paths))
    if hasattr(cache, "set_headwise_select_prune_config"):
        cache.set_headwise_select_prune_config(select_mode=HEAD_SELECT_MODE)
    if hasattr(cache, "set_headwise_diversity_prune_config"):
        cache.set_headwise_diversity_prune_config(candidate_ratio=HEAD_DIV_CANDIDATE_RATIO)
    if hasattr(cache, "set_headwise_dpp_prune_config"):
        cache.set_headwise_dpp_prune_config(
            score_alpha=HEAD_DPP_SCORE_ALPHA,
            diag_jitter=HEAD_DPP_DIAG_JITTER,
            enable_diagnostics=True,
        )

    (prune_result, prune_time) = measure_latency(
        cache.prune,
        PRUNE_RATIO,
        PRUNE_MODE,
        args.prune_exec,
    )
    (pruned_output, pruned_gen) = measure_latency(
        generate_from_prefill_cache,
        model,
        processor,
        prefill_inputs,
        cache,
        build_query_tail_text(args.final_question),
        args.max_new_tokens,
        COMPARE_GENERATE_KWARGS,
        torch.float16,
        decode_generated,
        False,
    )

    print(f"[Loaded Frames] count={len(frame_paths)}")
    print(format_selected_qa_report(selected_score_tasks))
    print(f"[Score] {summarize_score_tensors(score_tensors)}")
    print(f"[Prune] {summarize_prune_result(prune_result)}")
    print(format_dpp_summary(prune_result))
    print(
        "[Latency] "
        f"prep_manual={prep_manual:.3f}s "
        f"fresh_gen={fresh_gen:.3f}s "
        f"prep_prefill={prep_prefill:.3f}s "
        f"prefill_cache={prefill_cache:.3f}s "
        f"score_forward={score_forward:.3f}s "
        f"prune={prune_time:.3f}s "
        f"pruned_gen={pruned_gen:.3f}s"
    )
    print(f"[Final Question] {args.final_question}")
    if args.print_response:
        print(f"[Fresh Output] {fresh_output}")
        print(f"[Pruned Output] {pruned_output}")


def main():
    args = parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
