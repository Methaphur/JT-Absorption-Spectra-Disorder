"""Overlay computed spectra (and an optional reference curve)."""

import os
from typing import Dict, List

import matplotlib

# Use a non-interactive backend when we won't show a window.
import numpy as np

from .config import Config


def overlay(
    results: List[Dict],
    cfg: Config,
    out_path: str,
    show: bool = True,
) -> str:
    """Plot each Nv's spectrum on one axis; save PNG. Returns the output path."""
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("Disorder-averaged absorption spectra (κ/ω = 2.2)")

    # Optional "without disorder" reference.
    ref = cfg.reference_file
    if ref and os.path.isfile(ref):
        data = np.loadtxt(ref, comments="#")
        E_ref, I_ref = data[:, 0], data[:, 1]
        if I_ref.max() > 0:
            I_ref = I_ref / I_ref.max()
        ax.plot(E_ref, I_ref, "b", label="Without disorder")

    for res in sorted(results, key=lambda r: r["Nv"]):
        ax.plot(res["E"], res["spectrum"], linewidth=1.5, label=f"Nv={res['Nv']}")

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity")
    ax.set_xlim(cfg.E_min, cfg.E_max)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    return out_path


def overlay_sigma(
    results: List[Dict],
    cfg: Config,
    out_path: str,
    show: bool = True,
) -> str:
    """Plot spectra for a single Nv across disorder strengths sigma. Returns path."""
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Nv = results[0]["Nv"] if results else "?"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f"Absorption vs disorder strength (Nv={Nv}, κ/ω = 2.2)")

    ref = cfg.reference_file
    if ref and os.path.isfile(ref):
        data = np.loadtxt(ref, comments="#")
        E_ref, I_ref = data[:, 0], data[:, 1]
        if I_ref.max() > 0:
            I_ref = I_ref / I_ref.max()
        ax.plot(E_ref, I_ref, "b", label="Without disorder")

    for res in sorted(results, key=lambda r: r["sigma"]):
        ax.plot(
            res["E"], res["spectrum"], linewidth=1.5, label=f"σ={res['sigma']:g} eV"
        )

    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel("Intensity")
    ax.set_xlim(cfg.E_min, cfg.E_max)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
    return out_path
