#!/usr/bin/env python3
"""Sweep Nv, compute disorder-averaged spectra, save each, and overlay them.

Examples
--------
    # full default sweep (Nv = 2..12)
    python main.py

    # quick smoke test
    python main.py --nv 2 3 --realizations 3 --no-show

    # recompute everything from scratch
    python main.py --force
"""

import argparse
import os

from tqdm import tqdm

from spectrum.config import Config
from spectrum.plotting import overlay
from spectrum.spectrum import compute_spectrum_for_Nv
from spectrum.storage import (
    dat_path,
    ensure_results_dir,
    exists,
    load_result,
    result_path,
    save_result,
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nv", type=int, nargs="+", default=None,
                   help="Nv values to compute (default: config NV_LIST, i.e. 2..12).")
    p.add_argument("--realizations", type=int, default=None,
                   help="Override number of disorder realizations.")
    p.add_argument("--force", action="store_true",
                   help="Recompute even if a saved result exists.")
    p.add_argument("--no-show", action="store_true",
                   help="Save the overlay figure without opening a window.")
    p.add_argument("--results-dir", default=None,
                   help="Override output directory (default: config results_dir).")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = Config()
    if args.nv is not None:
        cfg.nv_list = args.nv
    if args.realizations is not None:
        cfg.n_realizations = args.realizations
    if args.results_dir is not None:
        cfg.results_dir = args.results_dir

    ensure_results_dir(cfg)

    results = []
    summary = []  # (Nv, status, dim)

    for Nv in tqdm(cfg.nv_list, desc="Nv sweep", unit="Nv"):
        if exists(Nv, cfg) and not args.force:
            res = load_result(Nv, cfg)
            results.append(res)
            summary.append((Nv, "loaded", res["dim"]))
            tqdm.write(f"[Nv={Nv}] loaded from {result_path(Nv, cfg)} (dim={res['dim']})")
            continue

        res = compute_spectrum_for_Nv(Nv, cfg, show_progress=True)
        save_result(res, cfg)  # persist immediately so progress survives a crash
        results.append(res)
        summary.append((Nv, "computed", res["dim"]))
        tqdm.write(
            f"[Nv={Nv}] computed & saved -> {result_path(Nv, cfg)} "
            f"(dim={res['dim']}, {cfg.n_realizations} realizations)"
        )

    # ---- Overlay plot ----
    out_fig = os.path.join(cfg.results_dir, "overlay_spectra.png")
    overlay(results, cfg, out_fig, show=not args.no_show)

    # ---- Summary ----
    print("\n=== Summary ===")
    for Nv, status, dim in summary:
        print(f"  Nv={Nv:>2}  {status:<8}  dim={dim:<6}  "
              f"{result_path(Nv, cfg)}  |  {dat_path(Nv, cfg)}")
    print(f"\nOverlay figure: {out_fig}")


if __name__ == "__main__":
    main()
