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
from spectrum.plotting import overlay, overlay_sigma
from spectrum.spectrum import compute_spectrum_for_Nv, compute_spectrum_sigma_sweep
from spectrum.storage import (
    dat_path,
    ensure_results_dir,
    exists,
    load_result,
    load_sigma_result,
    result_path,
    save_result,
    save_sigma_result,
    sigma_dat_path,
    sigma_exists,
    sigma_result_path,
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
    # ---- sigma-sweep mode (fixed Nv, multiple disorder strengths) ----
    p.add_argument("--sigma-sweep", action="store_true",
                   help="Sweep disorder strength sigma for a single Nv instead of "
                        "sweeping Nv.")
    p.add_argument("--sigmas", type=float, nargs="+", default=None,
                   help="Sigma values for --sigma-sweep (default: config sigma_list).")
    return p.parse_args()


def run_sigma_sweep(args, cfg: Config) -> None:
    """Compute spectra for one Nv across several disorder strengths and overlay."""
    # Nv for the sweep: first of --nv if given, else config default.
    Nv = args.nv[0] if args.nv else cfg.sigma_sweep_nv
    sigmas = args.sigmas if args.sigmas is not None else cfg.sigma_list

    ensure_results_dir(cfg)

    # Resume: figure out which sigmas still need computing.
    to_compute = [s for s in sigmas if args.force or not sigma_exists(Nv, s, cfg)]
    results = []
    summary = []  # (sigma, status)

    # Load any already-saved sigmas first.
    for s in sigmas:
        if s not in to_compute:
            res = load_sigma_result(Nv, s, cfg)
            results.append(res)
            summary.append((s, "loaded"))
            tqdm.write(f"[Nv={Nv} sigma={s:g}] loaded from {sigma_result_path(Nv, s, cfg)}")

    # Compute the rest (static H built once inside, reused across these sigmas).
    if to_compute:
        computed = compute_spectrum_sigma_sweep(Nv, to_compute, cfg, show_progress=True)
        for res in computed:
            save_sigma_result(res, cfg)
            results.append(res)
            summary.append((res["sigma"], "computed"))
            tqdm.write(
                f"[Nv={Nv} sigma={res['sigma']:g}] computed & saved -> "
                f"{sigma_result_path(Nv, res['sigma'], cfg)} (dim={res['dim']})"
            )

    out_fig = os.path.join(cfg.results_dir, f"overlay_sigma_Nv{Nv}.png")
    overlay_sigma(results, cfg, out_fig, show=not args.no_show)

    print("\n=== Sigma-sweep summary (Nv={}) ===".format(Nv))
    for s, status in sorted(summary):
        print(f"  sigma={s:<6g}  {status:<8}  {sigma_result_path(Nv, s, cfg)}  |  "
              f"{sigma_dat_path(Nv, s, cfg)}")
    print(f"\nOverlay figure: {out_fig}")


def main():
    args = parse_args()

    cfg = Config()
    if args.nv is not None:
        cfg.nv_list = args.nv
    if args.realizations is not None:
        cfg.n_realizations = args.realizations
    if args.results_dir is not None:
        cfg.results_dir = args.results_dir

    if args.sigma_sweep:
        run_sigma_sweep(args, cfg)
        return

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
