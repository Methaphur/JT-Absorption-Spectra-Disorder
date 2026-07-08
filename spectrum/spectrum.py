"""Disorder averaging and Lorentzian broadening.

Two entry points:

* :func:`compute_spectrum_for_Nv` — one spectrum for a single ``Nv`` at the
  configured ``sigma`` (used by the Nv sweep).
* :func:`compute_spectrum_sigma_sweep` — several spectra for a single ``Nv``,
  one per disorder strength ``sigma``. The static Hamiltonian does not depend
  on ``sigma``, so it is built once and reused across the whole sweep.
"""

from typing import Dict, List, Sequence, Tuple

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


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------
def _prepare_Nv(Nv: int, cfg: Config, show_progress: bool):
    """Build the sector basis and disorder-independent Hamiltonian for one Nv.

    Returns ``(H, mask1, mask2, i0, dim)``.
    """
    basis_final, basis_index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
    dim = len(basis_final)
    if dim == 0:
        raise ValueError(
            f"Empty sector for Nv={Nv} (Nex={cfg.nex_target}, Jz={cfg.jz_target})."
        )
    if cfg.initial_state not in basis_index:
        raise ValueError(
            f"Initial state {cfg.initial_state} not in the Nv={Nv} sector basis."
        )
    i0 = basis_index[cfg.initial_state]

    H = build_static_hamiltonian(Nv, basis_final, basis_index, cfg, show_progress)
    mask1, mask2 = electronic_masks(basis_final)
    return H, mask1, mask2, i0, dim


def _disorder_average(
    H: np.ndarray,
    mask1: np.ndarray,
    mask2: np.ndarray,
    i0: int,
    sigma: float,
    cfg: Config,
    show_progress: bool,
    desc: str,
    n_realizations: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average over disorder realizations at a given ``sigma``.

    The electronic diagonal is added in place before ``eigh`` and restored
    afterwards, so ``H`` is left unchanged for reuse across sigmas. Pass
    ``n_realizations`` to override ``cfg.n_realizations`` (used by the
    realization sweep). Because the RNG is seeded once, the first ``n`` rows of
    a longer run are identical to an independent ``n``-realization run.
    """
    n_real = cfg.n_realizations if n_realizations is None else n_realizations
    dim = H.shape[0]
    d = np.arange(dim)
    rng = np.random.default_rng(cfg.rng_seed)

    all_evals = np.empty((n_real, dim))
    all_intensity = np.empty((n_real, dim))

    iterator = range(n_real)
    if show_progress:
        iterator = tqdm(iterator, desc=desc, leave=False, unit="real")

    for r in iterator:
        eps1_dis = cfg.eps1 + rng.normal(0, sigma)
        eps2_dis = cfg.eps2 + rng.normal(0, sigma)
        elec = electronic_diagonal(mask1, mask2, eps1_dis, eps2_dis)

        H[d, d] += elec
        evals, evecs = np.linalg.eigh(H)
        H[d, d] -= elec

        intensity = np.abs(evecs[i0, :]) ** 2
        imax = intensity.max()
        if imax > 0:
            intensity = intensity / imax

        all_evals[r] = evals
        all_intensity[r] = intensity

    return all_evals, all_intensity


def _broaden(all_evals: np.ndarray, all_intensity: np.ndarray, cfg: Config):
    """Lorentzian-broaden and disorder-average onto the energy grid."""
    E = np.linspace(cfg.E_min, cfg.E_max, cfg.E_points)
    spectrum = np.zeros_like(E)
    for evals, intensity in zip(all_evals, all_intensity):
        for En, I in zip(evals, intensity):
            spectrum += I * lorentzian(E, En, cfg.gamma)
    smax = spectrum.max()
    if smax > 0:
        spectrum = spectrum / smax
    return E, spectrum


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def compute_spectrum_for_Nv(Nv: int, cfg: Config, show_progress: bool = True) -> Dict:
    """Disorder-averaged absorption spectrum for one ``Nv`` at ``cfg.sigma``."""
    H, mask1, mask2, i0, dim = _prepare_Nv(Nv, cfg, show_progress)

    all_evals, all_intensity = _disorder_average(
        H, mask1, mask2, i0, cfg.sigma, cfg, show_progress, f"  disorder(Nv={Nv})"
    )
    E, spectrum = _broaden(all_evals, all_intensity, cfg)

    return {
        "Nv": Nv,
        "sigma": cfg.sigma,
        "dim": dim,
        "E": E,
        "spectrum": spectrum,
        "all_evals": all_evals,
        "all_intensity": all_intensity,
    }


def compute_spectrum_sigma_sweep(
    Nv: int,
    sigma_list: Sequence[float],
    cfg: Config,
    show_progress: bool = True,
) -> List[Dict]:
    """Compute one spectrum per disorder strength ``sigma`` for a fixed ``Nv``.

    The static Hamiltonian is built once and reused for every ``sigma``; only
    the disorder loop reruns. Returns a list of result dicts (one per sigma),
    each shaped like :func:`compute_spectrum_for_Nv`'s output plus a ``sigma``
    key.
    """
    H, mask1, mask2, i0, dim = _prepare_Nv(Nv, cfg, show_progress)

    sigmas = list(sigma_list)
    outer = sigmas
    if show_progress:
        outer = tqdm(sigmas, desc=f"sigma sweep(Nv={Nv})", unit="sigma")

    results: List[Dict] = []
    for sigma in outer:
        all_evals, all_intensity = _disorder_average(
            H, mask1, mask2, i0, sigma, cfg, show_progress, f"  disorder(sigma={sigma:g})"
        )
        E, spectrum = _broaden(all_evals, all_intensity, cfg)
        results.append(
            {
                "Nv": Nv,
                "sigma": sigma,
                "dim": dim,
                "E": E,
                "spectrum": spectrum,
                "all_evals": all_evals,
                "all_intensity": all_intensity,
            }
        )

    return results


def compute_spectrum_realization_sweep(
    Nv: int,
    realization_list: Sequence[int],
    cfg: Config,
    show_progress: bool = True,
) -> List[Dict]:
    """Compute one spectrum per disorder-realization count for a fixed ``Nv``.

    Shows how the disorder average converges with the number of realizations.
    The disorder loop runs **once** at ``max(realization_list)`` and each smaller
    count reuses the leading rows (identical to an independent shorter run since
    the RNG is seeded once). Returns a list of result dicts, one per realization
    count, each with an ``n_realizations`` key.
    """
    H, mask1, mask2, i0, dim = _prepare_Nv(Nv, cfg, show_progress)

    counts = sorted(set(int(n) for n in realization_list))
    n_max = counts[-1]

    all_evals, all_intensity = _disorder_average(
        H, mask1, mask2, i0, cfg.sigma, cfg, show_progress,
        f"  disorder(Nv={Nv}, n={n_max})", n_realizations=n_max,
    )

    results: List[Dict] = []
    for n in counts:
        E, spectrum = _broaden(all_evals[:n], all_intensity[:n], cfg)
        results.append(
            {
                "Nv": Nv,
                "sigma": cfg.sigma,
                "n_realizations": n,
                "dim": dim,
                "E": E,
                "spectrum": spectrum,
                "all_evals": all_evals[:n],
                "all_intensity": all_intensity[:n],
            }
        )

    return results
