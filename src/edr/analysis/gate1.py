"""Ablation-stage (Gate 1) mechanical tally. NO narrative verdicts.

Preregistered criteria:
- A5: adjudication field = heldout ALL-158 paired bootstrap CI (the field
  follows the hypothesis's scope). scaffold-only 47 co-reported. Asymmetric
  branch: all-158 tied but scaffold-only significantly worse -> STRATUM_LIMITED.
- A6: frontier as-is; best cell enters, all reported; cells beating A2's full
  injection flagged as patch-interference evidence.
- A11: heldout is the main surface; union over scramble seeds; Wilson lower
  bound > 0 -> point-estimate deduction with CI attached; else "≈0".
- A7/A8: descriptive; A8's 100% point = the frozen round-1 5-seed reference,
  every point annotated with its seed count.

Every 3-seed number is tagged PROBE. Human sign-off required for interpretation.

CLI: python -m edr.analysis.gate1 [--output-dir outputs/ablations]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from edr.analysis.stats import paired_bootstrap, per_episode_success, wilson_ci
from edr.config import load_config
from edr.io_utils import read_json, write_json
from edr.paths import ABLATION_OUT, REFERENCE_EVALS, TEACHER_HELDOUT_REFERENCE


def stratum_of(heldout_payloads):
    strata = {}
    for payload in heldout_payloads:
        for record in payload["records"]:
            strata[record["episode_id"]] = record.get("partition")
    return strata


def load_glob(pattern, base):
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


def compare_arms(name, arm_payloads, reference_payloads, report):
    if not arm_payloads:
        report[name] = {"status": "MISSING_RUNS"}
        return
    arm = per_episode_success(arm_payloads)
    reference = per_episode_success(reference_payloads)
    common = sorted(set(arm) & set(reference))
    strata = stratum_of(reference_payloads)
    all_ci = paired_bootstrap(reference, arm, common)  # positive mean = reference (round-1 main) better
    scaffold_only = [e for e in common if strata.get(e) == "scaffold_only"]
    so_ci = paired_bootstrap(reference, arm, scaffold_only)
    entry = {
        "probe_label": f"PROBE signal (seeds={len(arm_payloads)})",
        "heldout": heldout_summary(arm_payloads),
        "reference_minus_arm_all": all_ci,
        "reference_minus_arm_scaffold_only": {**so_ci, "probe": f"PROBE signal (n={so_ci['n']})" if so_ci["n"] < 30 else None},
    }
    if name == "a5":
        entry["adjudication_field"] = "heldout all-158 (the field follows the hypothesis's scope)"
        if all_ci.get("ci_excludes_zero") and all_ci["mean_diff"] > 0:
            entry["mechanical_verdict"] = "A5_SIGNIFICANTLY_WORSE (all-158) -> verifier necessity supported"
        elif all_ci.get("ci_excludes_zero"):
            entry["mechanical_verdict"] = "A5_SIGNIFICANTLY_BETTER (all-158) -> honest finding, wording downgrade"
        elif so_ci.get("ci_excludes_zero") and (so_ci.get("mean_diff") or 0) > 0:
            entry["mechanical_verdict"] = (
                "STRATUM_LIMITED: all-158 not separated but scaffold-only significantly worse "
                "-> verifier necessity stratum-limited (preregistered branch)"
            )
        else:
            entry["mechanical_verdict"] = "NOT_SEPARATED -> honest finding, wording downgrade"
    report[name] = entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(ABLATION_OUT))
    args = parser.parse_args(argv)
    config = load_config("ablations")
    output_dir = Path(args.output_dir)

    report = {
        "probe": "gate1_mechanical_tally",
        "banner": "MECHANICAL TALLY ONLY — interpretation and gate verdicts require human sign-off.",
    }

    reference_heldout = load_glob("main/seed*/heldout.json", REFERENCE_EVALS)
    reference_sibling = load_glob("main/seed*/sibling.json", REFERENCE_EVALS)
    report["round1_main_reference"] = {
        "heldout": heldout_summary(reference_heldout),
        "sibling": sibling_summary(reference_sibling),
    }

    a2_repair = None
    if TEACHER_HELDOUT_REFERENCE.exists():
        teacher_ref = read_json(TEACHER_HELDOUT_REFERENCE)
        stats = teacher_ref.get("stats") or {}
        records = teacher_ref.get("records") or []
        a2_repair = stats.get("teacher_success")
        if a2_repair is None:
            a2_repair = sum(1 for r in records if (r.get("teacher") or {}).get("success") or r.get("success"))
        report["teacher_reference"] = {"heldout_repair": a2_repair, "heldout_n": stats.get("total", len(records))}
    else:
        report["teacher_reference"] = {"status": "MISSING_FROZEN_REFERENCE"}

    for arm in ("a5", "a7"):
        heldout_payloads = load_glob(f"{arm}/eval/*heldout*.json", output_dir)
        compare_arms(arm, heldout_payloads, reference_heldout, report)
        sib = sibling_summary(load_glob(f"{arm}/eval/*sibling*.json", output_dir))
        if isinstance(report[arm], dict):
            report[arm]["sibling"] = sib
    if isinstance(report.get("a7"), dict) and report["a7"].get("status") != "MISSING_RUNS":
        report["a7"]["definition_note"] = "A7 = recipe minus replay minus KL (the KL anchor rides on replay rows)"

    a8 = {}
    for fraction in config["a8"]["fractions"]:
        tag = f"{int(fraction * 100)}"
        payloads = load_glob(f"a8/eval/*{tag}pct*heldout*.json", output_dir)
        sib = sibling_summary(load_glob(f"a8/eval/*{tag}pct*sibling*.json", output_dir))
        a8[f"{tag}pct"] = {
            "heldout": heldout_summary(payloads) or {"status": "MISSING_RUNS"},
            "sibling": sib,
            "probe_label": f"PROBE signal (seeds={len(payloads)})" if payloads else None,
        }
    a8["100pct"] = {
        "heldout": heldout_summary(reference_heldout),
        "sibling": sibling_summary(reference_sibling),
        "note": "round-1 main artifacts reused unshaved (no retrain); seed counts differ by design",
    }
    report["a8"] = a8

    a6_summary_path = output_dir / "a6" / "a6_summary.json"
    if a6_summary_path.exists():
        a6 = read_json(a6_summary_path)
        cells = a6["cells"]
        report["a6"] = {
            "cells": cells,
            "best_cell": a6.get("best_cell"),
            "frontier_points": {
                "teacher_full_injection": {"repair": a2_repair, "ctx_tokens": "~12500 (round-1 measurement)"},
                "a6_best": cells.get(a6.get("best_cell"), {}),
                "round1_main_zero_context": {
                    "repair_mean": report["round1_main_reference"]["heldout"]["repair_mean"],
                    "ctx_tokens": 0,
                },
            },
        }
        if a2_repair is not None:
            interference = [name for name, cell in cells.items() if cell["repair"] > a2_repair]
            report["a6"]["patch_interference_flag"] = {
                "cells_exceeding_full_injection": interference,
                "note": "Preregistered observation: top-k beating full injection = direct patch-interference evidence.",
            }
    else:
        report["a6"] = {"status": "MISSING_RUNS"}

    a11_summary_path = output_dir / "a11" / "a11_summary.json"
    if a11_summary_path.exists():
        a11 = read_json(a11_summary_path)
        surfaces = {}
        for name, item in a11["surfaces"].items():
            ci = wilson_ci(item["placebo_repair"], item["n"])
            role = "MAIN adjudication surface" if name == "heldout" else "reference surface"
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
                "rule": "net gain = repair(teacher) − placebo union; POINT estimate is the deduction magnitude, CI attached",
            }
        report["a11"] = {"scramble_seeds": a11.get("scramble_seeds"), "main_surface": "heldout", "surfaces": surfaces}
    else:
        report["a11"] = {"status": "MISSING_RUNS"}

    write_json(output_dir / "gate1_report.json", report)
    write_markdown(output_dir / "gate1_report.md", report)
    print(f"[gate1] report -> {output_dir / 'gate1_report.md'}")


def write_markdown(path, report):
    ref = report["round1_main_reference"]
    lines = [
        "# Ablation-stage (Gate 1) mechanical tally",
        "",
        f"> {report['banner']}",
        "",
        "## References",
        f"- round-1 main (5 seeds): heldout repair mean = {_fmt(ref['heldout'])}",
        f"- round-1 main sibling forget mean = {ref['sibling']['forget_sibling_mean']:.4f}" if ref["sibling"] else "- sibling reference: missing",
        f"- teacher (scaffold-on) heldout repair = {report['teacher_reference'].get('heldout_repair')}/{report['teacher_reference'].get('heldout_n')}",
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
            lines.append(f"- paired bootstrap reference−{arm.upper()} (all): {entry.get('reference_minus_arm_all')}")
            lines.append(f"- paired bootstrap reference−{arm.upper()} (scaffold-only): {entry.get('reference_minus_arm_scaffold_only')}")
            if entry.get("sibling"):
                lines.append(f"- sibling forget mean: {entry['sibling']['forget_sibling_mean']:.4f}")
            if entry.get("mechanical_verdict"):
                lines.append(f"- mechanical verdict: **{entry['mechanical_verdict']}**")
            if entry.get("definition_note"):
                lines.append(f"- note: {entry['definition_note']}")
        lines.append("")
    lines.append("## A8 data-scale")
    lines.append("(100% = round-1 main's 5 seeds unshaved; seed counts differ by design — every point annotated)")
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
            lines.append(f"- patch-interference cells (> full injection): {flag.get('cells_exceeding_full_injection')}")
    lines.append("")
    lines.append("## A11 placebo-patch")
    a11 = report.get("a11", {})
    if a11.get("status") == "MISSING_RUNS":
        lines.append("- MISSING RUNS")
    else:
        lines.append(f"(union over scramble seeds {a11.get('scramble_seeds')}; main surface = heldout)")
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
    lines.append("All ablation arms are 3-seed: every number above is `PROBE signal` until upgraded to 5 seeds for headline use.")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt(summary):
    if not summary or summary.get("status"):
        return "missing"
    return f"{summary['repair_mean']:.1f}/{summary['heldout_n']} (rate {summary['repair_rate_mean']:.4f}, seeds={summary['seeds']})"


def _sib(sib):
    return f"{sib['forget_sibling_mean']:.4f}" if sib else "missing"


if __name__ == "__main__":
    main()
