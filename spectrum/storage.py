"""Per-Nv result persistence and resume/skip support."""

import os
from typing import Dict, Optional

import numpy as np

from .config import Config


def ensure_results_dir(cfg: Config) -> None:
    os.makedirs(cfg.results_dir, exist_ok=True)


def result_path(Nv: int, cfg: Config) -> str:
    return os.path.join(cfg.results_dir, f"spectrum_Nv{Nv}.npz")


def dat_path(Nv: int, cfg: Config) -> str:
    return os.path.join(cfg.results_dir, f"spectrum_Nv{Nv}.dat")


def exists(Nv: int, cfg: Config) -> bool:
    return os.path.isfile(result_path(Nv, cfg))


def save_result(result: Dict, cfg: Config) -> None:
    """Save a single Nv result: compressed .npz + a portable two-column .dat."""
    ensure_results_dir(cfg)
    Nv = int(result["Nv"])

    np.savez_compressed(
        result_path(Nv, cfg),
        Nv=Nv,
        dim=int(result["dim"]),
        E=result["E"],
        spectrum=result["spectrum"],
        all_evals=result["all_evals"],
        all_intensity=result["all_intensity"],
        # provenance
        n_realizations=cfg.n_realizations,
        sigma=cfg.sigma,
        gamma=cfg.gamma,
        rng_seed=cfg.rng_seed,
        eps1=cfg.eps1,
        eps2=cfg.eps2,
        omega=cfg.omega,
        kappa=cfg.kappa,
        omega_c=cfg.omega_c,
        Omega=cfg.Omega,
    )

    np.savetxt(
        dat_path(Nv, cfg),
        np.column_stack([result["E"], result["spectrum"]]),
        header=f"Nv={Nv}  dim={result['dim']}  n_real={cfg.n_realizations}\nE(eV)    intensity",
    )


def load_result(Nv: int, cfg: Config) -> Optional[Dict]:
    """Load a previously saved Nv result, or None if absent."""
    path = result_path(Nv, cfg)
    if not os.path.isfile(path):
        return None
    data = np.load(path)
    return {
        "Nv": int(data["Nv"]),
        "dim": int(data["dim"]),
        "E": data["E"],
        "spectrum": data["spectrum"],
        "all_evals": data["all_evals"],
        "all_intensity": data["all_intensity"],
    }
