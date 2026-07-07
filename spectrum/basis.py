"""Basis construction and symmetry-sector filtering.

A basis state is the 8-tuple ``(e1, np1, nm1, e2, np2, nm2, cp, cm)``:
electronic label of each molecule (``'A'``, ``'Ep'``, ``'Em'``), the b+/b-
vibrational occupations of each molecule, and the two cavity photon occupations.
"""

from itertools import product
from typing import Dict, List, Tuple

State = Tuple

ELECTRONIC = ("A", "Ep", "Em")


def build_full_basis(Nv: int) -> List[State]:
    """Full product basis with ``Nv`` vibrational levels per mode.

    Cavity photon occupations are restricted to {0, 1} (two photon modes).
    """
    basis: List[State] = []
    for e1, e2 in product(ELECTRONIC, ELECTRONIC):
        for np1, nm1 in product(range(Nv), range(Nv)):
            for np2, nm2 in product(range(Nv), range(Nv)):
                for cp, cm in product([0, 1], [0, 1]):
                    basis.append((e1, np1, nm1, e2, np2, nm2, cp, cm))
    return basis


def exctn_no(state: State) -> int:
    """Total excitation number Nex (electronic excitations + cavity photons)."""
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    Nex = 0
    if e1 != "A":
        Nex += 1
    if e2 != "A":
        Nex += 1
    Nex += cp + cm
    return Nex


def Jz(state: State) -> int:
    """Total angular-momentum-like quantum number Jz = 2*Lz + Sz + lz."""
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    Lz = (np1 - nm1) + (np2 - nm2)
    Sz = 0
    if e1 == "Ep":
        Sz += 1
    elif e1 == "Em":
        Sz -= 1
    if e2 == "Ep":
        Sz += 1
    elif e2 == "Em":
        Sz -= 1
    lz = cp - cm
    return (2 * Lz) + Sz + lz


def build_sector_basis(
    Nv: int, nex: int, jz: int
) -> Tuple[List[State], Dict[State, int]]:
    """Return the (basis_final, basis_index) for the given Nex and Jz sector."""
    full = build_full_basis(Nv)
    basis_final = [s for s in full if exctn_no(s) == nex and Jz(s) == jz]
    basis_index = {state: i for i, state in enumerate(basis_final)}
    return basis_final, basis_index
