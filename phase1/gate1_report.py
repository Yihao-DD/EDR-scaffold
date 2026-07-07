#!/usr/bin/env python3
"""Gate 1 mechanical tally. NO narrative verdicts — human sign-off required.

Preregistered Gate 1 criteria (PROJECT_MASTER_PLAN Part VI, v1.26/v1.27 rulings):
- A5: adjudication field = heldout ALL-158 paired bootstrap CI (the verifier
  hypothesis acts on ALL training data, so the adjudication field follows the
  hypothesis's scope — v1.27 principle). scaffold-only 47 is co-reported, not
  adjudicating. Preregistered asymmetric branch: all-158 not separated but
  scaffold-only significantly worse -> "verifier necessity STRATUM-LIMITED",
  Gate 1 partial, philosophy wording narrowed accordingly.
- A6: frontier reported as-is. Best cell enters the frontier; all cells
  reported. Preregistered extra observation: any cell repairing MORE than the
  A2 full-injection reference is flagged as patch-interference evidence.
- A11 (v1.27 operationalization): main surface = heldout 158; union over the
  three scramble seeds; Wilson 95% lower bound > 0 -> significantly above
  zero -> deduction fires (net gain = repair(A2) − placebo, POINT estimate,
  CI attached); lower bound ≤ 0 -> "≈0" holds, exclusion statement stands.
  train/val surfaces are reference-only.
- A7/A8: descriptive tables (replay necessity, data-scale curve). A8's 100%
  point = A3's original 5-seed artifacts, unshaved; every curve point is
  annotated with its seed count (v1.27).

Every phase-1 number is 3-seed -> tagged PROBE. Strata with n<30 tagged.
Output: gate1_report.json + gate1_report.md.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from phase1.common import (
    add_common_args,
    load_config,
    read_json,
    write_json,
    REPRO_ROOT,
)

BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 20260713


def per_episode_success(heldout_payloads):
    """episode_id -> mean success across seeds for one arm."""

    totals = {}
    counts = {}
    for payload in heldout_payloads:
        for record in payload["records"]:
            episode_id = record["episode_id"]
            totals[episode_id] = totals.get(episode_id, 0) + (1 if record["success"] else 0)
            counts[episode_id] = counts.get(episode_id, 0) + 1
    return {episode_id: totals[episode_id] / counts[episode_id] for episode_id in totals}


def stratum_of(heldout_payloads):
    strata = {}
    for payload in heldout_payloads:
        for record in payload["records"]:
            strata[record["episode_id"]] = record.get("partition")
    return strata


def paired_bootstrap(left, right, episode_ids, iterations=BOOTSTRAP_ITERATIONS, seed=BOOTSTRAP_SEED):
    """Bootstrap CI of mean(left - right) over episodes. Pure python, deterministic."""

    diffs = [left[e] - right[e] for e in episode_ids]
    n = len(diffs)
    if n == 0:
        return {"n": 0, "mean_diff": None, "ci95": None}
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * iterations)]
    hi = means[int(0.975 * iterations) - 1]
    return {
        "n": n,
        "mean_diff": sum(diffs) / n,
        "ci95": [lo, hi],
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
    }


def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def load_arm_heldout(pattern, base):
    return [read_json(path) for path in sorted(Path(base).glob(pattern))]


def load_arm_sibling(pattern, base):
    return [read_json(path) for path in sorted(Path(base).glob(pattern))]


def sibling_summary(payloads):
    if not payloads:
        return None
    rates = [payload["summary"]["forget_sibling"] for payload in payloads]
    succ = [payload["summary"]["success_rate"] for payload in payloads]
    return {
        "seeds": len(payloads),
        "forget_sibling_mean": sum(rates) / len(rates),
        "forget_sibling_per_seed": rates,
        "success_rate_mean": sum(succ) / len(succ),
    }


def heldout_summary(payloads):
    if not payloads:
        return None
    repairs = [payload["summary"]["heldout_repair"] for payload in payloads]
    n = payloads[0]["summary"]["heldout_n"]
    return {
        "seeds": len(payloads),
        "heldout_n": n,
        "repair_mean": sum(repairs) / len(repairs),
        "repair_per_seed": repairs,
        "repair_rate_mean": (sum(repairs) / len(repairs)) / n if n else None,
    }


def compare_arms(name, arm_payloads, a3_payloads, report):
    if not arm_payloads:
        report[name] = {"status": "MISSING_RUNS"}
        return
    arm = per_episode_success(arm_payloads)
    a3 = per_episode_success(a3_payloads)
    common = sorted(set(arm) & set(a3))
    strata = stratum_of(a3_payloads)
    all_ci = paired_bootstrap(a3, arm, common)  # positive mean = A3 better
    scaffold_only = [e for e in common if strata.get(e) == "scaffold_only"]
    so_ci = paired_bootstrap(a3, arm, scaffold_only)
    entry = {
        "probe_label": f"PROBE signal (seeds={len(arm_payloads)})",
        "heldout": heldout_summary(arm_payloads),
        "a3_minus_arm_all": all_ci,
        "a3_minus_arm_scaffold_only": {**so_ci, "probe": so_ci["n"] < 30 and f"PROBE signal (n={so_ci['n']})" or None},
    }
    if name == "a5":
        entry["adjudication_field"] = "heldout all-158 (v1.27: the field follows the hypothesis's scope)"
        if all_ci.get("ci_excludes_zero") and all_ci["mean_diff"] > 0:
            entry["mechanical_verdict"] = "A5_SIGNIFICANTLY_WORSE_THAN_A3 (all-158) -> verifier necessity supported (Gate 1)"
        elif all_ci.get("ci_excludes_zero"):
            entry["mechanical_verdict"] = "A5_SIGNIFICANTLY_BETTER_THAN_A3 (all-158) -> honest finding, philosophy layer-4 wording downgrade"
        elif so_ci.get("ci_excludes_zero") and (so_ci.get("mean_diff") or 0) > 0:
            entry["mechanical_verdict"] = (
                "STRATUM_LIMITED: all-158 not separated but scaffold-only significantly worse "
                "-> verifier necessity stratum-limited, Gate 1 partial (v1.27 preregistered branch)"
            )
        else:
            entry["mechanical_verdict"] = "NOT_SEPARATED -> honest finding, philosophy layer-4 wording downgrade"
    report[name] = entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    output_dir = Path(args.output_dir)

    report = {
        "probe": "gate1_mechanical_tally",
        "banner": "MECHANICAL TALLY ONLY — Gate interpretation and narrative require human sign-off (Anti-self-deception protocol #7).",
    }

    # A3 reference: the five rep2 artifacts.
    a3_heldout = [read_json(p) for p in sorted((REPRO_ROOT / "artifacts").glob("main_rep2_seed*/heldout.json"))]
    a3_sibling = [read_json(p) for p in sorted((REPRO_ROOT / "artifacts").glob("main_rep2_seed*/sibling.json"))]
    report["a3_reference"] = {
        "heldout": heldout_summary(a3_heldout),
        "sibling": sibling_summary(a3_sibling),
    }

    # A2 reference: frozen teacher re-anchor heldout forward (50/158).
    teacher_ref_path = REPRO_ROOT / config["eval"]["teacher_heldout_reference"]
    a2_repair = None
    if teacher_ref_path.exists():
        teacher_ref = read_json(teacher_ref_path)
        stats = teacher_ref.get("stats") or {}
        records = teacher_ref.get("records") or []
        a2_repair = stats.get("teacher_success")
        if a2_repair is None:
            a2_repair = sum(1 for r in records if (r.get("teacher") or {}).get("success") or r.get("success"))
        report["a2_reference"] = {"heldout_repair": a2_repair, "heldout_n": stats.get("total", len(records)), "source": str(teacher_ref_path)}
    else:
        report["a2_reference"] = {"status": "MISSING_FROZEN_REFERENCE", "expected": str(teacher_ref_path)}

    # --- A5 / A7 -------------------------------------------------------
    for arm in ("a5", "a7"):
        heldout_payloads = load_arm_heldout(f"{arm}/eval/*heldout*.json", output_dir)
        compare_arms(arm, heldout_payloads, a3_heldout, report)
        sib = sibling_summary(load_arm_sibling(f"{arm}/eval/*sibling*.json", output_dir))
        if isinstance(report[arm], dict):
            report[arm]["sibling"] = sib
    if isinstance(report.get("a7"), dict) and report["a7"].get("status") != "MISSING_RUNS":
        report["a7"]["definition_note"] = "A7 = rep2 minus replay minus KL (KL rides on replay rows; v1.26)"

    # --- A8 --------------------------------------------------------------
    a8 = {}
    for fraction in config["phase1"]["a8"]["fractions"]:
        tag = f"{int(fraction * 100)}"
        payloads = load_arm_heldout(f"a8/eval/*{tag}pct*heldout*.json", output_dir)
        sib = sibling_summary(load_arm_sibling(f"a8/eval/*{tag}pct*sibling*.json", output_dir))
        a8[f"{tag}pct"] = {
            "heldout": heldout_summary(payloads) or {"status": "MISSING_RUNS"},
            "sibling": sib,
            "probe_label": f"PROBE signal (seeds={len(payloads)})" if payloads else None,
        }
    a8["100pct"] = {"heldout": heldout_summary(a3_heldout), "sibling": sibling_summary(a3_sibling), "note": "A3 rep2 artifacts reused (no retrain)"}
    report["a8"] = a8

    # --- A6 --------------------------------------------------------------
    a6_summary_path = output_dir / "a6" / "a6_summary.json"
    if a6_summary_path.exists():
        a6 = read_json(a6_summary_path)
        cells = a6["cells"]
        report["a6"] = {
            "cells": cells,
            "best_cell": a6.get("best_cell"),
            "frontier_points": {
                "A2_full_injection": {"repair": a2_repair, "ctx_tokens": "~12500 (Phase 0 measurement)"},
                "A6_best": cells.get(a6.get("best_cell"), {}),
                "A3_zero_context": {"repair_mean": report["a3_reference"]["heldout"]["repair_mean"], "ctx_tokens": 0},
            },
        }
        if a2_repair is not None:
            interference = [name for name, cell in cells.items() if cell["repair"] > a2_repair]
            report["a6"]["patch_interference_flag"] = {
                "cells_exceeding_A2": interference,
                "note": "Preregistered observation (v1.26): top-k beating full injection = direct patch-interference evidence.",
            }
    else:
        report["a6"] = {"status": "MISSING_RUNS"}

    # --- A11 -------------------------------------------------------------
    a11_summary_path = output_dir / "a11" / "a11_summary.json"
    if a11_summary_path.exists():
        a11 = read_json(a11_summary_path)
        surfaces = {}
        for name, item in a11["surfaces"].items():
            ci = wilson_ci(item["placebo_repair"], item["n"])
            role = "MAIN adjudication surface (v1.27)" if name == "heldout" else "reference surface"
            surfaces[name] = {
                **item,
                "role": role,
                "wilson_ci95_union_rate": ci,
                "mechanical_verdict": "SIGNIFICANTLY_ABOVE_ZERO -> deduction fires" if ci[0] > 0 else "≈0 -> perturbation explanation excluded",
            }
        if a2_repair is not None and "heldout" in surfaces:
            item = surfaces["heldout"]
            ci = item["wilson_ci95_union_rate"]
            item["net_teaching_gain_heldout"] = {
                "point": a2_repair - item["placebo_repair"],
                "ci95": [a2_repair - item["n"] * ci[1], a2_repair - item["n"] * ci[0]],
                "rule": "net gain = repair(A2) − placebo union; POINT estimate is the deduction magnitude, CI attached (v1.27)",
            }
        report["a11"] = {"scramble_seeds": a11.get("scramble_seeds"), "main_surface": "heldout", "surfaces": surfaces}
    else:
        report["a11"] = {"status": "MISSING_RUNS"}

    write_json(output_dir / "gate1_report.json", report)
    write_markdown(output_dir / "gate1_report.md", report)
    print(f"[gate1] report -> {output_dir / 'gate1_report.md'}")


def write_markdown(path, report):
    lines = [
        "# Gate 1 mechanical tally",
        "",
        f"> {report['banner']}",
        "",
        "## A3 / A2 references",
        f"- A3 (rep2, 5 seeds): heldout repair mean = {_fmt(report['a3_reference']['heldout'])}",
        f"- A3 sibling forget mean = {report['a3_reference']['sibling']['forget_sibling_mean']:.4f}" if report["a3_reference"]["sibling"] else "- A3 sibling: missing",
        f"- A2 (scaffold-on) heldout repair = {report['a2_reference'].get('heldout_repair')}/{report['a2_reference'].get('heldout_n')}",
        "",
    ]
    for arm in ("a5", "a7"):
        entry = report.get(arm, {})
        lines.append(f"## {arm.upper()}")
        if entry.get("status") == "MISSING_RUNS":
            lines.append("- MISSING RUNS")
        else:
            lines.append(f"- {entry.get('probe_label')}")
            lines.append(f"- heldout: {_fmt(entry.get('heldout'))}")
            lines.append(f"- paired bootstrap A3−{arm.upper()} (all): {entry.get('a3_minus_arm_all')}")
            lines.append(f"- paired bootstrap A3−{arm.upper()} (scaffold-only): {entry.get('a3_minus_arm_scaffold_only')}")
            if entry.get("sibling"):
                lines.append(f"- sibling forget mean: {entry['sibling']['forget_sibling_mean']:.4f}")
            if entry.get("mechanical_verdict"):
                lines.append(f"- mechanical verdict: **{entry['mechanical_verdict']}**")
            if entry.get("definition_note"):
                lines.append(f"- note: {entry['definition_note']}")
        lines.append("")
    lines.append("## A8 data-scale")
    lines.append("(v1.27: 100% = A3's original 5 seeds unshaved; seed counts differ by design — every point annotated, error bars speak)")
    for tag, item in report.get("a8", {}).items():
        seeds = (item.get("heldout") or {}).get("seeds", "?")
        lines.append(f"- {tag} (seeds={seeds}): heldout={_fmt(item.get('heldout'))} sibling_forget={_sib(item.get('sibling'))} {item.get('note', '')}")
    lines.append("")
    lines.append("## A6 retrieval-patch")
    a6 = report.get("a6", {})
    if a6.get("status") == "MISSING_RUNS":
        lines.append("- MISSING RUNS")
    else:
        for name, cell in sorted(a6.get("cells", {}).items()):
            lines.append(f"- {name}: repair={cell['repair']}/{cell['heldout_n']} ctx_mean={cell['ctx_tokens_mean']:.0f}")
        lines.append(f"- best cell: {a6.get('best_cell')}")
        flag = a6.get("patch_interference_flag", {})
        if flag:
            lines.append(f"- patch-interference cells (> A2): {flag.get('cells_exceeding_A2')}")
    lines.append("")
    lines.append("## A11 placebo-patch")
    a11 = report.get("a11", {})
    if a11.get("status") == "MISSING_RUNS":
        lines.append("- MISSING RUNS")
    else:
        lines.append(f"(union over scramble seeds {a11.get('scramble_seeds')}; main surface = heldout — v1.27)")
        for name, item in a11.get("surfaces", {}).items():
            lines.append(
                f"- {name} [{item.get('role', '')}]: union placebo_repair={item['placebo_repair']}/{item['n']} "
                f"per_seed={item.get('per_seed_repair')} wilson95={item['wilson_ci95_union_rate']} -> {item['mechanical_verdict']}"
            )
        gain = (a11.get("surfaces", {}).get("heldout") or {}).get("net_teaching_gain_heldout")
        if gain:
            lines.append(f"- net teaching gain (heldout): point={gain['point']} ci95={gain['ci95']}")
    lines.append("")
    lines.append("---")
    lines.append("All phase-1 arms are 3-seed: every number above is `PROBE signal` until upgraded to 5 seeds for headline use.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(summary):
    if not summary or summary.get("status"):
        return "missing"
    return f"{summary['repair_mean']:.1f}/{summary['heldout_n']} (rate {summary['repair_rate_mean']:.4f}, seeds={summary['seeds']})"


def _sib(sib):
    return f"{sib['forget_sibling_mean']:.4f}" if sib else "missing"


if __name__ == "__main__":
    main()
