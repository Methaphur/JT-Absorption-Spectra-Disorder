"""Central configuration: all physical and run parameters.

Values are taken verbatim from the notebook. The electronic energies ``eps1`` /
``eps2`` were only present in a commented-out notebook cell; they are defined
here so the Hamiltonian is fully specified.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Config:
    # ---------------- Physical parameters (eV) ----------------
    omega: float = 0.08196          # vibrational quantum
    kappa: float = 0.180312         # Jahn-Teller (vibronic) coupling
    omega_c: float = 6.85           # cavity photon energy
    Omega: float = 2 * 6.85 * 0.05  # collective light-matter coupling scale

    N: int = 2                      # number of molecules
    Np: int = 2                     # cavity photon-number cutoff (states 0, 1)

    # Electronic site energies: eps_i = eps + delta_i
    eps: float = 7.0
    delta1: float = 0.2
    delta2: float = -0.2

    # ---------------- Disorder ----------------
    sigma: float = 0.03             # disorder strength W (eV)
    n_realizations: int = 20
    rng_seed: int = 1234

    # ---------------- Spectrum grid / broadening ----------------
    E_min: float = 6.0
    E_max: float = 8.0
    E_points: int = 2000
    gamma: float = 0.044            # Lorentzian FWHM (eV)

    # ---------------- Basis / sector selection ----------------
    initial_state: Tuple = ("A", 0, 0, "A", 0, 0, 0, 1)
    nex_target: int = 1             # single-excitation subspace
    jz_target: int = -1             # Jz symmetry sector

    # ---------------- Nv sweep ----------------
    nv_list: List[int] = field(default_factory=lambda: list(range(2, 13)))

    # ---------------- Paths ----------------
    results_dir: str = "results"
    # Optional "without disorder" reference curve (two columns: E, intensity).
    # Left as None: the overlay simply skips it when absent.
    reference_file: Optional[str] = None

    # ---------------- Derived ----------------
    @property
    def eps1(self) -> float:
        return self.eps + self.delta1

    @property
    def eps2(self) -> float:
        return self.eps + self.delta2

    @property
    def g(self) -> float:
        """Per-molecule matter-cavity coupling."""
        import numpy as np
        return self.Omega / (2 * np.sqrt(self.N))


# A ready-to-use default instance.
default_config = Config()
