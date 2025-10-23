import jaxley as jx
from jaxley_mech.channels.aoyama00 import Ca, Kar, Kdr, Kto, Leak, Na


def build_HC():
    """Build a single horizontal cell with the Aoyama 2000 channels."""
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=1)
    HC = jx.Cell([branch], parents=[-1])

    HC.compute_xyz()

    # Insert the mechanisms
    HC.insert(Kdr(name="HC_Kdr"))
    HC.insert(Kto(name="HC_Kto"))
    HC.insert(Kar(name="HC_Kar"))
    HC.insert(Ca(name="HC_Ca"))
    HC.insert(Leak(name="HC_Leak"))
    HC.insert(Na(name="HC_Na"))

    HC.set("HC_Ca_gCa", 3.3e-3)  # nS (replicates Aoyama plots)

    HC.init_states()

    return HC
