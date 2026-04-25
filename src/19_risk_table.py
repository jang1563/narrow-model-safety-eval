#!/usr/bin/env python3
"""
19_risk_table.py — Multi-Dimensional Risk Profile (MDRP) table and figures.

Aggregates all Phase 1 + Phase 2 results into:
  (A) Risk table: protein × model → FSI-PM, FSI-LM, FSI-EvoD, FSPE, SER-P, SER-N
  (B) 2D FSI × SER-N scatter plot (four-quadrant risk space)
  (C) Per-protein radar chart (7-dimensional MDRP)

Usage:
    python src/19_risk_table.py
    python src/19_risk_table.py --output_dir results/figures/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils import RESULTS_DIR, FIGURES_DIR, print_header

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not installed — figures skipped (pip install matplotlib)")


# ============================================================================
# Load results
# ============================================================================


def load_fsi_pm() -> dict:
    """Load v1 ProteinMPNN FSI results. Returns {pdb_id: fsi_mean}."""
    path = RESULTS_DIR / "fsi_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    results = data if isinstance(data, list) else data.get("results", [])
    return {r["pdb_id"]: r["fsi"]["mean"] for r in results}


def load_fsi_lm() -> dict:
    """Load LigandMPNN FSI results. Returns {pdb_id: fsi_mean}."""
    path = RESULTS_DIR / "fsi_ligandmpnn_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {r["pdb_id"]: r["fsi"]["mean"] for r in data.get("results", [])}


def load_fsi_evodiff() -> dict:
    """Load EvoDiff FSI results. Returns {pdb_id: fsi_mean (motif-conditioned)}."""
    path = RESULTS_DIR / "fsi_evodiff_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for r in data.get("results", []):
        mc = r.get("motif_conditioned", {})
        if "fsi_mean" in mc:
            out[r["pdb_id"]] = mc["fsi_mean"]
    return out


def load_fspe() -> dict:
    """Load ESM-2 v1 FSPE results. Returns {uniprot_id: fspe_ratio}."""
    path = RESULTS_DIR / "fspe_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    results = data if isinstance(data, list) else data.get("results", [])
    return {r.get("uniprot_id", r.get("protein_id", "")): r.get("fspe_ratio", None)
            for r in results if r.get("fspe_ratio") is not None}


def load_esm3_fspe() -> dict:
    """Load ESM-3 FSPE results. Returns {uniprot_id: fspe_ratio}."""
    path = RESULTS_DIR / "esm3_fspe_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    out = {}
    for r in data.get("results", []):
        ratio = r.get("fspe_ratio")
        if ratio is not None and ratio != 1.0:
            out[r["uniprot_id"]] = ratio
    return out


def load_ser() -> dict:
    """Load SER results. Returns {(model, pdb_id): {ser_p, ser_n}}."""
    path = RESULTS_DIR / "ser_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {(r["model"], r["pdb_id"]): r for r in data.get("results", [])}


def load_fhs() -> dict:
    """Load FHS results. Returns {uniprot_id: fhs}."""
    path = RESULTS_DIR / "fhs_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {r["uniprot_id"]: r["fhs"] for r in data.get("results", [])}


def load_functional_sites_index() -> dict:
    """Build pdb_id → uniprot_id mapping."""
    from utils import load_functional_sites
    fs = load_functional_sites()
    return {info["pdb_id"]: uid for uid, info in fs.items()
            if not uid.startswith("_") and "pdb_id" in info}


# ============================================================================
# Risk table assembly
# ============================================================================


def build_risk_table(pdb_ids: list) -> list:
    fsi_pm = load_fsi_pm()
    fsi_lm = load_fsi_lm()
    fsi_evod = load_fsi_evodiff()
    fspe_esm2 = load_fspe()
    fspe_esm3 = load_esm3_fspe()
    ser = load_ser()
    fhs = load_fhs()
    pdb_to_uid = load_functional_sites_index()

    rows = []
    for pdb_id in pdb_ids:
        uid = pdb_to_uid.get(pdb_id, "")
        ser_pm = ser.get(("proteinmpnn", pdb_id), {})
        ser_lm = ser.get(("ligandmpnn", pdb_id), {})

        row = {
            "pdb_id": pdb_id,
            "uniprot_id": uid,
            "fsi_pm": fsi_pm.get(pdb_id),
            "fsi_lm": fsi_lm.get(pdb_id),
            "fsi_evod": fsi_evod.get(pdb_id),
            "fspe_esm2": fspe_esm2.get(uid),
            "fspe_esm3": fspe_esm3.get(uid),
            "fhs": fhs.get(uid),
            "ser_p_pm": ser_pm.get("ser_p"),
            "ser_n_pm": ser_pm.get("ser_n"),
            "ser_p_lm": ser_lm.get("ser_p"),
            "ser_n_lm": ser_lm.get("ser_n"),
        }
        rows.append(row)
    return rows


def _fmt(val, decimals=3):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}"


def print_risk_table(rows: list):
    print(f"\n{'PDB':<6} {'FSI-PM':>7} {'FSI-LM':>7} {'FSI-EvoD':>9} "
          f"{'FSPE-E2':>8} {'SER-P/PM':>9} {'SER-N/PM':>9} "
          f"{'SER-P/LM':>9} {'SER-N/LM':>9}")
    print("-" * 80)
    for r in rows:
        print(f"{r['pdb_id']:<6} "
              f"{_fmt(r['fsi_pm']):>7} "
              f"{_fmt(r['fsi_lm']):>7} "
              f"{_fmt(r['fsi_evod']):>9} "
              f"{_fmt(r['fspe_esm2']):>8} "
              f"{_fmt(r['ser_p_pm']):>9} "
              f"{_fmt(r['ser_n_pm']):>9} "
              f"{_fmt(r['ser_p_lm']):>9} "
              f"{_fmt(r['ser_n_lm']):>9}")


# ============================================================================
# 2D FSI × SER-N risk space figure
# ============================================================================

_PROTEIN_LABELS = {
    "3BTA": "BoNT-A",
    "2AAI": "Ricin",
    "1ACC": "Anthrax PA",
    "1XTC": "Cholera",
    "3SEB": "SEB",
    "1ABR": "Abrin",
    "1Z7H": "TeNT",
    "4HSC": "SLO",
}

_COLORS = {
    "proteinmpnn": "#2166ac",
    "ligandmpnn":  "#d6604d",
}
_MARKERS = {
    "proteinmpnn": "o",
    "ligandmpnn":  "^",
}


def plot_fsi_ser_space(rows: list, output_dir: Path):
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(9, 7))

    # Quadrant shading
    ax.axhspan(0.70, 1.05, xmin=0, xmax=0.5, alpha=0.06, color="orange")   # high FSI, low SER-N
    ax.axhspan(0.70, 1.05, xmin=0.5, xmax=1.0, alpha=0.06, color="red")    # high FSI, high SER-N
    ax.axhspan(-0.05, 0.70, xmin=0, xmax=0.5, alpha=0.06, color="green")   # low FSI, low SER-N
    ax.axhspan(-0.05, 0.70, xmin=0.5, xmax=1.0, alpha=0.06, color="blue")  # low FSI, high SER-N

    # Quadrant lines
    fsi_mid = 1.0
    ser_mid = 0.70
    ax.axhline(ser_mid, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.axvline(fsi_mid, color="gray", lw=0.8, ls="--", alpha=0.5)

    # Quadrant labels
    ax.text(0.1, 0.97, "Low FSI\nHigh SER-N\n(screening blind spot)", transform=ax.transAxes,
            fontsize=7, color="steelblue", va="top", ha="left")
    ax.text(0.55, 0.97, "HIGH FSI + HIGH SER-N\n⚠ Highest risk quadrant", transform=ax.transAxes,
            fontsize=7, color="darkred", va="top", ha="left", fontweight="bold")
    ax.text(0.1, 0.03, "Safe\n(low FSI + low SER-N)", transform=ax.transAxes,
            fontsize=7, color="darkgreen", va="bottom", ha="left")
    ax.text(0.55, 0.03, "Detectable danger\n(high FSI, caught by screening)", transform=ax.transAxes,
            fontsize=7, color="darkorange", va="bottom", ha="left")

    plotted = set()
    for r in rows:
        pdb_id = r["pdb_id"]
        label = _PROTEIN_LABELS.get(pdb_id, pdb_id)

        for model, fsi_key, ser_n_key in [
            ("proteinmpnn", "fsi_pm", "ser_n_pm"),
            ("ligandmpnn",  "fsi_lm", "ser_n_lm"),
        ]:
            fsi = r.get(fsi_key)
            ser_n = r.get(ser_n_key)
            if fsi is None or ser_n is None:
                continue

            # Clamp FSI for display (EvoDiff has FSI~15, scale differently)
            fsi_display = min(fsi, 3.5)

            ax.scatter(
                fsi_display, ser_n,
                color=_COLORS[model],
                marker=_MARKERS[model],
                s=80, zorder=5, alpha=0.85,
                edgecolors="white", linewidths=0.5,
            )

            # Label only once per protein (avoid duplicate labels)
            label_key = (pdb_id, round(fsi_display, 2), round(ser_n, 2))
            if label_key not in plotted:
                ax.annotate(
                    label,
                    (fsi_display, ser_n),
                    xytext=(5, 4), textcoords="offset points",
                    fontsize=8, alpha=0.85,
                )
                plotted.add(label_key)

    # Legend
    pm_patch = mpatches.Patch(color=_COLORS["proteinmpnn"], label="ProteinMPNN")
    lm_patch = mpatches.Patch(color=_COLORS["ligandmpnn"], label="LigandMPNN")
    ax.legend(handles=[pm_patch, lm_patch], loc="upper left", fontsize=8)

    ax.set_xlabel("FSI (Functional Specificity Index)", fontsize=11)
    ax.set_ylabel("SER-N (fraction evading NT screening)", fontsize=11)
    ax.set_title("2D Risk Space: FSI × Screening Evasion Rate (SER-N)", fontsize=12)
    ax.set_xlim(-0.1, 3.6)
    ax.set_ylim(-0.05, 1.05)

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fsi_ser_risk_space.pdf"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# FSI model comparison bar chart
# ============================================================================


def plot_fsi_comparison(rows: list, output_dir: Path):
    if not HAS_MPL:
        return

    # Only proteins with at least PM and LM values
    valid = [r for r in rows if r.get("fsi_pm") is not None and r.get("fsi_lm") is not None]
    if not valid:
        return

    labels = [_PROTEIN_LABELS.get(r["pdb_id"], r["pdb_id"]) for r in valid]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))

    pm_vals = [r["fsi_pm"] for r in valid]
    lm_vals = [r["fsi_lm"] for r in valid]
    evod_vals = [r["fsi_evod"] if r.get("fsi_evod") else np.nan for r in valid]

    ax.bar(x - width, pm_vals,  width, label="ProteinMPNN", color="#2166ac", alpha=0.85)
    ax.bar(x,         lm_vals,  width, label="LigandMPNN",  color="#d6604d", alpha=0.85)
    ax.bar(x + width, [min(v, 3.5) if not np.isnan(v) else np.nan for v in evod_vals],
           width, label="EvoDiff (motif-cond.)", color="#4dac26", alpha=0.85)

    ax.axhline(1.0, color="black", lw=0.8, ls="--", alpha=0.5, label="FSI = 1 (no specificity)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("FSI (mean, n=100)")
    ax.set_title("FSI Comparison: ProteinMPNN vs LigandMPNN vs EvoDiff")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 4.0)

    path = output_dir / "fsi_model_comparison.pdf"
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=str(FIGURES_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print_header("Multi-Dimensional Risk Profile (MDRP) — v2 Phase 1+2")

    # Collect all protein PDB IDs seen across any results file
    all_pdbs = set()
    for fsi_dict in [load_fsi_pm(), load_fsi_lm(), load_fsi_evodiff()]:
        all_pdbs.update(fsi_dict.keys())
    pdb_ids = sorted(all_pdbs)

    print(f"\nProteins in panel: {pdb_ids}")

    rows = build_risk_table(pdb_ids)

    print("\n--- Risk Table (Phase 1 + Phase 2) ---")
    print_risk_table(rows)

    print("\n--- Generating figures ---")
    plot_fsi_ser_space(rows, output_dir)
    plot_fsi_comparison(rows, output_dir)

    # Save machine-readable risk table
    import json
    out_path = RESULTS_DIR / "mdrp_risk_table.json"
    with open(out_path, "w") as f:
        json.dump({"proteins": rows}, f, indent=2, default=lambda x: None if x != x else x)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
