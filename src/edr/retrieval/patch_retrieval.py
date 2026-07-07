"""A6 retrieval-patch baseline: BM25 + pinned dense retriever, k in {1,3}.

Preregistered spec:
- Two retrievers, four cells (bm25/dense x k=1/k=3); the best cell enters the
  context-cost frontier, all four are reported.
- Query = the episode's user instruction + the candidate function NAME list
  (no schemas). Doc = each accepted H1 patch, full text.
- No query expansion, no reranker, no LLM anywhere in the retrieval path.
- The index and per-episode retrieval results are frozen to disk with hashes;
  evaluation reads the frozen retrieval only. Dense model name + resolved
  revision are pinned into the frozen artifact.
- Preregistered observation: any cell repairing MORE than A2 (full injection)
  is direct patch-interference evidence, reported separately.

Subcommands:
  build-index   freezes retrieval (dense encoding may use GPU).
  eval          GPU forward over heldout 158 per cell; repair + measured ctx_cost.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path

from edr.config import load_config, resolve_model_id
from edr.data.splits import load_round1_inputs
from edr.io_utils import read_json, sha256_text, write_json
from edr.paths import ABLATION_OUT, HELDOUT_PARTITION, resolve
from edr.scaffold.patches import build_prompt
from edr.scaffold.teacher import parse_prediction, success_for_prediction

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text):
    return [token.lower() for token in TOKEN_PATTERN.findall(text or "")]


class BM25:
    """Plain Okapi BM25 (k1=1.5, b=0.75). Pure python, deterministic, no deps."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc) for doc in docs]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avg_len = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        df = Counter()
        for freqs in self.doc_freqs:
            for term in freqs:
                df[term] += 1
        n = len(self.doc_tokens)
        self.idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def scores(self, query):
        query_tokens = tokenize(query)
        out = []
        for freqs, length in zip(self.doc_freqs, self.doc_len):
            score = 0.0
            for term in query_tokens:
                if term not in freqs:
                    continue
                tf = freqs[term]
                denom = tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1.0))
                score += self.idf.get(term, 0.0) * tf * (self.k1 + 1) / denom
            out.append(score)
        return out


def build_query(episode):
    names = [
        function.get("name", "")
        for function in episode.get("function_pool", [])
        if isinstance(function, dict)
    ]
    return f"{episode.get('query', '')}\nCandidate functions: {', '.join(names)}"


def top_k(scores, k):
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return order[:k]


def load_heldout_episodes(baseline):
    partition = read_json(HELDOUT_PARTITION)
    baseline_by_id = {row["episode_id"]: row for row in baseline["records"]}
    episodes = []
    for row in partition["episodes"]:
        episode = baseline_by_id.get(row["episode_id"])
        if episode is None:
            raise AssertionError({"assert": "a6_heldout_in_baseline", "episode_id": row["episode_id"]})
        episodes.append((row, episode))
    return episodes


def dense_encode(texts, model_name, revision, batch_size=32):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, revision=revision)
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    try:
        from huggingface_hub import model_info

        resolved = model_info(model_name, revision=revision).sha
    except Exception:
        resolved = revision or "unresolved"
    return embeddings, resolved


def build_index(args, config):
    baseline, _evolution, _split, patches = load_round1_inputs(config["split_seed"])
    episodes = load_heldout_episodes(baseline)
    if args.limit:
        episodes = episodes[: args.limit]

    docs = [patch.patch_text for patch in patches]
    patch_ids = [patch.patch_id for patch in patches]
    queries = [build_query(episode) for _, episode in episodes]
    a6 = config["a6"]
    k_values = a6["k_values"]

    retrieval = {}
    meta = {
        "n_patches": len(patches),
        "n_queries": len(queries),
        "doc_hash": sha256_text("\n".join(docs)),
        "query_hash": sha256_text("\n".join(queries)),
        "k_values": k_values,
        "query_spec": "user instruction + candidate function name list (no schemas)",
    }

    if "bm25" in a6["retrievers"]:
        bm25 = BM25(docs)
        cells = {}
        for k in k_values:
            per_episode = {}
            for (row, episode), query in zip(episodes, queries):
                idx = top_k(bm25.scores(query), k)
                per_episode[episode["episode_id"]] = [patch_ids[i] for i in idx]
            cells[str(k)] = per_episode
        retrieval["bm25"] = cells
        meta["bm25"] = {"k1": 1.5, "b": 0.75, "tokenizer": "regex [A-Za-z0-9_]+ lowercase"}

    if "dense" in a6["retrievers"]:
        if args.dry_run:
            meta["dense"] = {"skipped": "dry-run does not encode"}
        else:
            doc_emb, resolved = dense_encode(docs, a6["dense_model"], a6.get("dense_revision"))
            query_emb, _ = dense_encode(queries, a6["dense_model"], a6.get("dense_revision"))
            sims = query_emb @ doc_emb.T
            cells = {}
            for k in k_values:
                per_episode = {}
                for qi, (row, episode) in enumerate(episodes):
                    idx = top_k(list(sims[qi]), k)
                    per_episode[episode["episode_id"]] = [patch_ids[i] for i in idx]
                cells[str(k)] = per_episode
            retrieval["dense"] = cells
            meta["dense"] = {"model": a6["dense_model"], "revision_requested": a6.get("dense_revision"), "revision_resolved": resolved}

    payload = {"probe": "a6_frozen_retrieval", "meta": meta, "retrieval": retrieval}
    output = Path(args.output_dir) / "a6" / "a6_frozen_retrieval.json"
    write_json(output, payload)
    print(f"[a6-index] retrievers={sorted(retrieval)} queries={len(queries)} patches={len(patches)} -> {output}")


def evaluate(args, config):
    baseline, _evolution, _split, patches = load_round1_inputs(config["split_seed"])
    patches_by_id = {patch.patch_id: patch for patch in patches}
    episodes = load_heldout_episodes(baseline)
    frozen = read_json(Path(args.output_dir) / "a6" / "a6_frozen_retrieval.json")
    retrieval = frozen["retrieval"]
    max_new_tokens = config["eval"]["max_new_tokens"]

    plan = []
    for retriever, cells in sorted(retrieval.items()):
        for k, per_episode in sorted(cells.items()):
            plan.append((retriever, k, per_episode))

    if args.dry_run:
        write_json(
            Path(args.output_dir) / "a6" / "a6_eval_plan.json",
            {
                "probe": "a6_eval_dry_run",
                "cells": [f"{retriever}:k{k}" for retriever, k, _ in plan],
                "heldout_n": len(episodes),
                "forwards_total": len(plan) * len(episodes),
            },
        )
        print(f"[a6-eval:dry-run] cells={len(plan)} forwards={len(plan) * len(episodes)}")
        return

    from edr.modeling import generate_one, load_model_for_eval, load_tokenizer

    model_id = resolve_model_id(config)
    tokenizer = load_tokenizer(model_id)
    model = load_model_for_eval(model_id, adapter_dir=None, qlora=not args.no_qlora)
    if args.limit:
        episodes = episodes[: args.limit]

    results = {}
    for retriever, k, per_episode in plan:
        records = []
        for index, (row, episode) in enumerate(episodes, start=1):
            if index == 1 or index % 25 == 0 or index == len(episodes):
                print(f"[a6-eval {retriever}:k{k}] {index}/{len(episodes)}", flush=True)
            retrieved = [patches_by_id[pid] for pid in per_episode.get(episode["episode_id"], []) if pid in patches_by_id]
            prompt = build_prompt(episode, retrieved)
            patch_context = "\n\n".join(f"[PATCH {i}] {p.patch_text}" for i, p in enumerate(retrieved, start=1))
            ctx_tokens = len(tokenizer(patch_context, add_special_tokens=False)["input_ids"]) if patch_context else 0
            raw = generate_one(model, tokenizer, prompt, max_new_tokens)
            prediction, parse_error = parse_prediction(raw)
            success = success_for_prediction(prediction, episode)
            records.append(
                {
                    "episode_id": episode["episode_id"],
                    "partition": row.get("partition"),
                    "success": success,
                    "prediction": prediction,
                    "raw_model_output": raw,
                    "parse_error": parse_error,
                    "retrieved_patch_ids": [p.patch_id for p in retrieved],
                    "ctx_tokens": ctx_tokens,
                }
            )
        repair = sum(1 for r in records if r["success"])
        cell = {
            "retriever": retriever,
            "k": int(k),
            "heldout_n": len(records),
            "repair": repair,
            "repair_rate": repair / len(records) if records else 0.0,
            "ctx_tokens_mean": sum(r["ctx_tokens"] for r in records) / len(records) if records else 0.0,
            "by_partition": _by_partition(records),
            "records": records,
        }
        results[f"{retriever}:k{k}"] = cell
        write_json(Path(args.output_dir) / "a6" / f"a6_eval_{retriever}_k{k}.json", cell)

    summary = {
        "probe": "a6_retrieval_eval",
        "model_id": model_id,
        "frozen_retrieval_meta": frozen["meta"],
        "cells": {
            name: {key: value for key, value in cell.items() if key != "records"}
            for name, cell in results.items()
        },
    }
    best = max(results.items(), key=lambda item: item[1]["repair"])
    summary["best_cell"] = best[0]
    write_json(Path(args.output_dir) / "a6" / "a6_summary.json", summary)
    for name, cell in sorted(results.items()):
        print(f"[a6-eval] {name}: repair={cell['repair']}/{cell['heldout_n']} ctx_mean={cell['ctx_tokens_mean']:.0f}")
    print(f"[a6-eval] best cell = {best[0]}")


def _by_partition(records):
    out = {}
    for value in sorted({r.get("partition") for r in records}):
        rows = [r for r in records if r.get("partition") == value]
        out[str(value)] = {"n": len(rows), "repair": sum(1 for r in rows if r["success"])}
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    index_parser = subparsers.add_parser("build-index")
    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--no-qlora", action="store_true")
    for sub in (index_parser, eval_parser):
        sub.add_argument("--output-dir", default=str(ABLATION_OUT))
        sub.add_argument("--limit", type=int, default=None)
        sub.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config("ablations")
    if args.command == "build-index":
        build_index(args, config)
    else:
        evaluate(args, config)


if __name__ == "__main__":
    main()
