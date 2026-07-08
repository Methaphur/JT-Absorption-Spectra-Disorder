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
# Optional parallel path (opt-in via cfg.n_workers > 1)
#
# The serial code above is untouched and remains the default. When enabled,
# disorder realizations run across worker processes. To keep results the same
# as serial, the disorder draws are generated once in the parent (in the same
# order as the serial loop) and scattered to the workers. Each worker is pinned
# to a single BLAS thread (a single dense ``eigh`` barely benefits from more),
# so N workers give roughly an N-fold speedup on independent realizations.
# ---------------------------------------------------------------------------
_WORKER: Dict = {}

# Environment variables that control BLAS / threading backends.
_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _parallel_enabled(cfg: Config) -> bool:
    n = getattr(cfg, "n_workers", None)
    return bool(n) and n > 1


def _worker_init(Nv: int, cfg: Config) -> None:
    """Build the disorder-independent Hamiltonian once per worker process."""
    basis_final, basis_index = build_sector_basis(Nv, cfg.nex_target, cfg.jz_target)
    H = build_static_hamiltonian(Nv, basis_final, basis_index, cfg, show_progress=False)
    mask1, mask2 = electronic_masks(basis_final)
    _WORKER["H"] = H
    _WORKER["mask1"] = mask1
    _WORKER["mask2"] = mask2
    _WORKER["i0"] = basis_index[cfg.initial_state]
    _WORKER["d"] = np.arange(H.shape[0])


def _worker_task(task):
    """Diagonalize one realization; mirrors the serial loop body exactly."""
    r, eps1_dis, eps2_dis = task
    H = _WORKER["H"]
    d = _WORKER["d"]
    elec = electronic_diagonal(_WORKER["mask1"], _WORKER["mask2"], eps1_dis, eps2_dis)

    H[d, d] += elec
    evals, evecs = np.linalg.eigh(H)
    H[d, d] -= elec

    intensity = np.abs(evecs[_WORKER["i0"], :]) ** 2
    imax = intensity.max()
    if imax > 0:
        intensity = intensity / imax
    return r, evals, intensity


def _disorder_average_parallel(
    Nv: int,
    sigma: float,
    cfg: Config,
    show_progress: bool,
    desc: str,
    n_realizations: int = None,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Parallel version of :func:`_disorder_average`. Returns (evals, I, dim).

    Disorder draws are generated in the parent in the same order as the serial
    loop, so the set of realizations is identical to the serial path.
    """
    import multiprocessing as mp
    import os
    from concurrent.futures import ProcessPoolExecutor

    n_real = cfg.n_realizations if n_realizations is None else n_realizations

    # Same RNG stream and draw order as the serial loop.
    rng = np.random.default_rng(cfg.rng_seed)
    tasks = []
    for r in range(n_real):
        eps1_dis = cfg.eps1 + rng.normal(0, sigma)
        eps2_dis = cfg.eps2 + rng.normal(0, sigma)
        tasks.append((r, eps1_dis, eps2_dis))

    # Pin BLAS to one thread per worker; spawned children inherit the env.
    saved = {k: os.environ.get(k) for k in _THREAD_ENV_VARS}
    for k in _THREAD_ENV_VARS:
        os.environ[k] = "1"

    all_evals = [None] * n_real
    all_intensity = [None] * n_real
    try:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=cfg.n_workers,
            mp_context=ctx,
            initializer=_worker_init,
            initargs=(Nv, cfg),
        ) as ex:
            iterator = ex.map(_worker_task, tasks, chunksize=1)
            if show_progress:
                iterator = tqdm(iterator, total=n_real, desc=desc, leave=False, unit="real")
            for r, evals, intensity in iterator:
                all_evals[r] = evals
                all_intensity[r] = intensity
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    dim = all_evals[0].shape[0]
    return np.array(all_evals), np.array(all_intensity), dim


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------
def compute_spectrum_for_Nv(Nv: int, cfg: Config, show_progress: bool = True) -> Dict:
    """Disorder-averaged absorption spectrum for one ``Nv`` at ``cfg.sigma``."""
    if _parallel_enabled(cfg):
        all_evals, all_intensity, dim = _disorder_average_parallel(
            Nv, cfg.sigma, cfg, show_progress, f"  disorder(Nv={Nv})"
        )
    else:
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
    parallel = _parallel_enabled(cfg)
    if not parallel:
        H, mask1, mask2, i0, dim = _prepare_Nv(Nv, cfg, show_progress)

    sigmas = list(sigma_list)
    outer = sigmas
    if show_progress:
        outer = tqdm(sigmas, desc=f"sigma sweep(Nv={Nv})", unit="sigma")

    results: List[Dict] = []
    for sigma in outer:
        if parallel:
            all_evals, all_intensity, dim = _disorder_average_parallel(
                Nv, sigma, cfg, show_progress, f"  disorder(sigma={sigma:g})"
            )
        else:
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
    counts = sorted(set(int(n) for n in realization_list))
    n_max = counts[-1]

    if _parallel_enabled(cfg):
        all_evals, all_intensity, dim = _disorder_average_parallel(
            Nv, cfg.sigma, cfg, show_progress,
            f"  disorder(Nv={Nv}, n={n_max})", n_realizations=n_max,
        )
    else:
        H, mask1, mask2, i0, dim = _prepare_Nv(Nv, cfg, show_progress)
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
