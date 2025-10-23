import jaxley as jx
from jaxley_mech.channels.chen24 import Phototransduction
from jaxley_mech.channels.kamiyama09 import (Ca, CaNernstReversal, CaPump,
                                             ClCa, Hyper, KCa, Kv, Leak)


def build_PR(ptc_params: dict = None):
    """Build a single photoreceptor."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=1)
    PR = jx.Cell([branch], parents=[-1])
    PR.compute_xyz()

    # Insert the mechanisms
    solver = "explicit"
    PR.insert(Kv(name="PR_Kv"))
    PR.insert(KCa(name="PR_KCa"))
    PR.insert(Hyper(name="PR_Hyper", solver=solver))
    PR.insert(ClCa(name="PR_ClCa"))
    PR.insert(Ca(name="PR_Ca"))
    PR.insert(CaPump(name="PR_CaPump", solver=solver))
    PR.insert(CaNernstReversal(name="PR_CaNernstReversal"))
    PR.insert(Leak(name="PR_Leak"))
    PR.insert(Phototransduction(name="PR_Phototransduction", solver=solver))

    # Set the PR cascade params
    if not ptc_params:
        # (params from jaxley-mech/Rieke initialization for mouse cones)
        ptc_params = {
            "PR_Phototransduction_sigma": 9.74,  # /s, opsin decay rate constant
            "PR_Phototransduction_phi": 9.74,  # /s, PDE decay rate constant
            "PR_Phototransduction_eta": 761.0,  # /s, PDE dark activate rate
            "PR_Phototransduction_gamma": 20.0,  # unitless, Opsin gain
            "PR_Phototransduction_beta": 2.64,  # /s, Ca2+ extrusion rate constant
            "PR_Phototransduction_k": 0.01,  # pA / μM, cGMP-to-current constant
            "PR_Phototransduction_m": 4.0,  # unitless, Cooperativity of GC Ca2+ dependence
            "PR_Phototransduction_n": 3.0,  # unitless, cGMP cahnnel cooperativity
            "PR_Phototransduction_K_GC": 0.4,  # μM, Affinity of GC Ca2+ dependence
            "PR_Phototransduction_G_dark": 20.0,  # μM, cGMP concentration in dark
            "PR_Phototransduction_C_dark": 1.0,  # μM, Ca2+ concentration in dark
        }
    for name, param in ptc_params.items():
        PR.set(name, param)

    PR.init_states()

    return PR, ptc_params
