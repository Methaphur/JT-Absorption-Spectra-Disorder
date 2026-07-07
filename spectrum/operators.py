"""Second-quantized operators acting on basis states.

Every operator takes a state tuple and returns ``(coeff, new_state)`` or
``None`` if it annihilates the state. Behavior is identical to the notebook;
the photon-number-aware cavity operators (the notebook's *second* definitions)
are the ones kept here.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Vibrational ladder operators (b+ / b- of molecules 1 and 2)
# ---------------------------------------------------------------------------
def bp1(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if np1 == 0:
        return None
    return np.sqrt(np1), (e1, np1 - 1, nm1, e2, np2, nm2, cp, cm)


def bp2(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if np2 == 0:
        return None
    return np.sqrt(np2), (e1, np1, nm1, e2, np2 - 1, nm2, cp, cm)


def bm1(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if nm1 == 0:
        return None
    return np.sqrt(nm1), (e1, np1, nm1 - 1, e2, np2, nm2, cp, cm)


def bm2(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if nm2 == 0:
        return None
    return np.sqrt(nm2), (e1, np1, nm1, e2, np2, nm2 - 1, cp, cm)


def bp1_dag(state, Nv):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if np1 == Nv - 1:
        return None
    return np.sqrt(np1 + 1), (e1, np1 + 1, nm1, e2, np2, nm2, cp, cm)


def bp2_dag(state, Nv):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if np2 == Nv - 1:
        return None
    return np.sqrt(np2 + 1), (e1, np1, nm1, e2, np2 + 1, nm2, cp, cm)


def bm1_dag(state, Nv):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if nm1 == Nv - 1:
        return None
    return np.sqrt(nm1 + 1), (e1, np1, nm1 + 1, e2, np2, nm2, cp, cm)


def bm2_dag(state, Nv):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if nm2 == Nv - 1:
        return None
    return np.sqrt(nm2 + 1), (e1, np1, nm1, e2, np2, nm2 + 1, cp, cm)


# ---------------------------------------------------------------------------
# Cavity photon operators (a+ / a-), photon-number-aware
# ---------------------------------------------------------------------------
def ap(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if cp == 0:
        return None
    return np.sqrt(cp), (e1, np1, nm1, e2, np2, nm2, cp - 1, cm)


def am(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if cm == 0:
        return None
    return np.sqrt(cm), (e1, np1, nm1, e2, np2, nm2, cp, cm - 1)


def ap_dag(state, Np):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if cp == Np - 1:
        return None
    return np.sqrt(cp + 1), (e1, np1, nm1, e2, np2, nm2, cp + 1, cm)


def am_dag(state, Np):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if cm == Np - 1:
        return None
    return np.sqrt(cm + 1), (e1, np1, nm1, e2, np2, nm2, cp, cm + 1)


# ---------------------------------------------------------------------------
# Electronic transition operators
# ---------------------------------------------------------------------------
def Ep_to_Em_1(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if e1 != "Ep":
        return None
    return 1.0, ("Em", np1, nm1, e2, np2, nm2, cp, cm)


def Em_to_Ep_1(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if e1 != "Em":
        return None
    return 1.0, ("Ep", np1, nm1, e2, np2, nm2, cp, cm)


def Ep_to_Em_2(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if e2 != "Ep":
        return None
    return 1.0, (e1, np1, nm1, "Em", np2, nm2, cp, cm)


def Em_to_Ep_2(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if e2 != "Em":
        return None
    return 1.0, (e1, np1, nm1, "Ep", np2, nm2, cp, cm)


def Ep_to_A_1(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if e1 != "Ep":
        return None
    return 1.0, ("A", np1, nm1, e2, np2, nm2, cp, cm)


def A_to_Ep_1(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if e1 != "A":
        return None
    return 1.0, ("Ep", np1, nm1, e2, np2, nm2, cp, cm)


def Em_to_A_1(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if e1 != "Em":
        return None
    return 1.0, ("A", np1, nm1, e2, np2, nm2, cp, cm)


def A_to_Em_1(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if e1 != "A":
        return None
    return 1.0, ("Em", np1, nm1, e2, np2, nm2, cp, cm)


def Ep_to_A_2(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if e2 != "Ep":
        return None
    return 1.0, (e1, np1, nm1, "A", np2, nm2, cp, cm)


def A_to_Ep_2(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if e2 != "A":
        return None
    return 1.0, (e1, np1, nm1, "Ep", np2, nm2, cp, cm)


def Em_to_A_2(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if e2 != "Em":
        return None
    return 1.0, (e1, np1, nm1, "A", np2, nm2, cp, cm)


def A_to_Em_2(state):
    e1, np1, nm1, e2, np2, nm2, cp, cm = state
    if e2 != "A":
        return None
    return 1.0, (e1, np1, nm1, "Em", np2, nm2, cp, cm)
