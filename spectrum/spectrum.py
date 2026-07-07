"""Disorder averaging and Lorentzian broadening for a single Nv."""

from typing import Dict

import numpy as np
from tqdm import tqdm

from .basis import build_sector_basis
from .config import Config
from .hamiltonian import (
    build_static_hamiltonian,
    electronic_diagonal,
    electronic_masks,
)


def lorentzian(x, x0, gamma):
    return (1 / np.pi) * (0.5 * gamma) / ((x - x0) ** 2 + (0.5 * gamma) ** 2)


def compute_spectrum_for_Nv(Nv: int, cfg: Config, show_progress: bool = True) -> Dict:
    """Compute the disorder-averaged absorption spectrum for one ``Nv``.

    Returns a dict with the energy grid, broadened spectrum, and the raw
    per-realization eigenvalues / intensities.
    """
    basis_final, basis_index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
    dim = len(basis_final)
    if dim == 0:
        raise ValueError(f"Empty sector for Nv={Nv} (Nex={cfg.nex_target}, Jz={cfg.jz_target}).")

    if cfg.initial_state not in basis_index:
        raise ValueError(
            f"Initial state {cfg.initial_state} not in the Nv={Nv} sector basis."
        )
    i0 = basis_index[cfg.initial_state]

    H = build_static_hamiltonian(Nv, basis_final, basis_index, cfg, show_progress)
    mask1, mask2 = electronic_masks(basis_final)
    d = np.arange(dim)

    rng = np.random.default_rng(cfg.rng_seed)

    all_evals = np.empty((cfg.n_realizations, dim))
    all_intensity = np.empty((cfg.n_realizations, dim))

    realizations = range(cfg.n_realizations)
    if show_progress:
        realizations = tqdm(
            realizations,
            desc=f"  disorder(Nv={Nv})",
            leave=False,
            unit="real",
        )

    for r in realizations:
        eps1_dis = cfg.eps1 + rng.normal(0, cfg.sigma)
        eps2_dis = cfg.eps2 + rng.normal(0, cfg.sigma)
        elec = electronic_diagonal(mask1, mask2, eps1_dis, eps2_dis)

        # Add electronic diagonal in place, diagonalize, then restore.
        H[d, d] += elec
        evals, evecs = np.linalg.eigh(H)
        H[d, d] -= elec

        intensity = np.abs(evecs[i0, :]) ** 2
        imax = intensity.max()
        if imax > 0:
            intensity = intensity / imax

        all_evals[r] = evals
        all_intensity[r] = intensity

    # ---- Lorentzian-broadened, disorder-averaged spectrum ----
    E = np.linspace(cfg.E_min, cfg.E_max, cfg.E_points)
    spectrum = np.zeros_like(E)
    for evals, intensity in zip(all_evals, all_intensity):
        for En, I in zip(evals, intensity):
            spectrum += I * lorentzian(E, En, cfg.gamma)
    smax = spectrum.max()
    if smax > 0:
        spectrum = spectrum / smax

    return {
        "Nv": Nv,
        "dim": dim,
        "E": E,
        "spectrum": spectrum,
        "all_evals": all_evals,
        "all_intensity": all_intensity,
    }
