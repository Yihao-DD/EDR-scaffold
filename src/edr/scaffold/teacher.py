"""Teacher forwards and sampling (scaffold-on generation).

Per-episode sampling seeds derive from stable_int with the frozen label
strings ("teacher", "pass16") — identical to round 1, so re-sampling
reproduces the original streams exactly.
"""

from __future__ import annotations

import time

from edr.io_utils import stable_int
from edr.scaffold.patches import build_prompt, relevant_patches_for_episode
from edr.verifier import call_success, parse_model_output


def parse_prediction(raw_text):
    try:
        return parse_model_output(raw_text), None
    except Exception as error:  # noqa: BLE001
        return None, str(error)


def success_for_prediction(prediction, episode):
    return call_success(prediction or {"name": "", "arguments": {}}, episode["ground_truth_call"])


def evaluate_teacher_episode(runner, episode, patches, max_new_tokens):
    relevant = relevant_patches_for_episode(patches, episode)
    if not relevant:
        return {
            "episode_id": episode["episode_id"],
            "success": False,
            "prediction": None,
            "raw_model_output": "",
            "parse_error": "no_relevant_patch",
            "patch_ids": [],
            "n_patches": 0,
        }
    prompt = build_prompt(episode, relevant)
    started = time.time()
    raw = runner.generate_many(prompt, n=1, do_sample=False, temperature=0.0, max_new_tokens=max_new_tokens)[0]
    latency_ms = int((time.time() - started) * 1000)
    prediction, parse_error = parse_prediction(raw)
    return {
        "episode_id": episode["episode_id"],
        "success": success_for_prediction(prediction, episode),
        "prediction": prediction,
        "raw_model_output": raw,
        "parse_error": parse_error,
        "latency_ms": latency_ms,
        "patch_ids": [patch.patch_id for patch in relevant],
        "n_patches": len(relevant),
    }


def sample_base_episode(runner, episode, prompt_template, sample_count, sample_batch_size, temperature, max_new_tokens, split_seed):
    from edr.data.splits import build_base_prompt

    prompt = build_base_prompt(episode, prompt_template)
    samples = []
    remaining = sample_count
    offset = 0
    started = time.time()
    while remaining:
        batch = min(sample_batch_size, remaining)
        seed = stable_int(split_seed, episode["episode_id"], "pass16", offset)
        raws = runner.generate_many(
            prompt,
            n=batch,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        for raw in raws:
            prediction, parse_error = parse_prediction(raw)
            success = success_for_prediction(prediction, episode)
            samples.append(
                {
                    "sample_index": offset,
                    "success": success,
                    "prediction": prediction,
                    "raw_model_output": raw,
                    "parse_error": parse_error,
                }
            )
            offset += 1
        remaining -= batch
    latency_ms = int((time.time() - started) * 1000)
    return {
        "episode_id": episode["episode_id"],
        "pass16_hits": sum(1 for sample in samples if sample["success"]),
        "sample_count": sample_count,
        "latency_ms": latency_ms,
        "successful_samples": [sample for sample in samples if sample["success"]],
    }


def sample_teacher_episode(runner, episode, patches, sample_count, sample_batch_size, temperature, max_new_tokens, split_seed):
    """Round-1 teacher augmentation: retains AST-passing samples only."""

    relevant = relevant_patches_for_episode(patches, episode)
    if not relevant:
        return {
            "episode_id": episode["episode_id"],
            "successful_samples": [],
            "sample_count": sample_count,
            "patch_ids": [],
            "error": "no_relevant_patch",
        }
    prompt = build_prompt(episode, relevant)
    samples = []
    remaining = sample_count
    offset = 0
    started = time.time()
    while remaining:
        batch = min(sample_batch_size, remaining)
        seed = stable_int(split_seed, episode["episode_id"], "teacher", offset)
        raws = runner.generate_many(
            prompt,
            n=batch,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        for raw in raws:
            prediction, parse_error = parse_prediction(raw)
            success = success_for_prediction(prediction, episode)
            samples.append(
                {
                    "sample_index": offset,
                    "success": success,
                    "prediction": prediction,
                    "raw_model_output": raw,
                    "parse_error": parse_error,
                }
            )
            offset += 1
        remaining -= batch
    return {
        "episode_id": episode["episode_id"],
        "sample_count": sample_count,
        "temperature": temperature,
        "latency_ms": int((time.time() - started) * 1000),
        "patch_ids": [patch.patch_id for patch in relevant],
        "successful_samples": [sample for sample in samples if sample["success"]],
    }


def sample_episode_keep_all(runner, episode, patches, sample_count, temperature, max_new_tokens, split_seed):
    """A5 variant: T=0 x1 plus T=temperature x sample_count, retaining EVERY
    sample with its verifier flag recorded (verified and unverified alike).
    Uses the same deterministic per-episode seeds as round 1."""

    relevant = relevant_patches_for_episode(patches, episode)
    prompt = build_prompt(episode, relevant)
    samples = []
    started = time.time()

    def record(raw, sample_index, kind):
        prediction, parse_error = parse_prediction(raw)
        success = success_for_prediction(prediction, episode)
        samples.append(
            {
                "sample_index": sample_index,
                "kind": kind,
                "success": success,
                "prediction": prediction,
                "raw_model_output": raw,
                "parse_error": parse_error,
            }
        )

    greedy = runner.generate_many(prompt, n=1, do_sample=False, temperature=0.0, max_new_tokens=max_new_tokens)[0]
    record(greedy, -1, "t0")
    offset = 0
    remaining = sample_count
    while remaining:
        batch = min(4, remaining)
        seed = stable_int(split_seed, episode["episode_id"], "teacher", offset)
        raws = runner.generate_many(
            prompt,
            n=batch,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        for raw in raws:
            record(raw, offset, "t_sampled")
            offset += 1
        remaining -= batch
    return {
        "episode_id": episode["episode_id"],
        "failure_class": episode.get("failure_class", "unknown"),
        "function_name": episode.get("ground_truth_call", {}).get("name", ""),
        "patch_ids": [patch.patch_id for patch in relevant],
        "n_patches": len(relevant),
        "sample_count": sample_count,
        "temperature": temperature,
        "latency_ms": int((time.time() - started) * 1000),
        "samples": samples,
        "ast_pass_count": sum(1 for sample in samples if sample["success"]),
    }
