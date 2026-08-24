from typing import Iterable

import torch
from time import perf_counter


def assert_same_text(expected: str, actual: str, label: str) -> None:
    if expected != actual:
        raise AssertionError(
            f"{label} mismatch.\nEXPECTED:\n{expected}\n\nACTUAL:\n{actual}"
        )


def assert_same_batch_features(
    left,
    right,
    label: str,
    atol: float = 1e-5,
    rtol: float = 1e-5,
    ignore_keys: Iterable[str] = (),
) -> None:
    ignore_keys = set(ignore_keys)
    left_keys = {key for key in left.keys() if key not in ignore_keys}
    right_keys = {key for key in right.keys() if key not in ignore_keys}
    if left_keys != right_keys:
        raise AssertionError(f"{label} keys differ: {sorted(left_keys)} != {sorted(right_keys)}")

    for key in sorted(left_keys):
        left_value = left[key]
        right_value = right[key]
        if torch.is_tensor(left_value):
            if not torch.is_tensor(right_value):
                raise AssertionError(f"{label}[{key}] type mismatch.")
            if left_value.shape != right_value.shape:
                raise AssertionError(f"{label}[{key}] shape mismatch: {left_value.shape} != {right_value.shape}")
            if left_value.dtype != right_value.dtype:
                raise AssertionError(f"{label}[{key}] dtype mismatch: {left_value.dtype} != {right_value.dtype}")
            if left_value.dtype.is_floating_point:
                same = torch.allclose(left_value, right_value, atol=atol, rtol=rtol)
            else:
                same = torch.equal(left_value, right_value)
            if not same:
                raise AssertionError(f"{label}[{key}] tensor values differ.")
        else:
            if left_value != right_value:
                raise AssertionError(f"{label}[{key}] values differ: {left_value} != {right_value}")


def assert_same_generated_ids(left_ids: torch.Tensor, right_ids: torch.Tensor, label: str) -> None:
    if left_ids.shape != right_ids.shape:
        raise AssertionError(f"{label} shape mismatch: {left_ids.shape} != {right_ids.shape}")
    if not torch.equal(left_ids, right_ids):
        raise AssertionError(f"{label} token ids differ.")


def print_sample_status(sample_id: str, status: str, detail: str | None = None) -> None:
    if detail:
        print(f"[{status}] {sample_id}: {detail}")
    else:
        print(f"[{status}] {sample_id}")


def print_summary(total: int, matched: int, mismatched: int, errors: int) -> None:
    print(
        "Summary: "
        f"total={total} matched={matched} mismatched={mismatched} errors={errors}"
    )


def cuda_sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_latency(func, *args, **kwargs):
    cuda_sync_if_needed()
    start = perf_counter()
    result = func(*args, **kwargs)
    cuda_sync_if_needed()
    return result, perf_counter() - start
