import jaxley as jx
from jaxley_mech.channels import benison01, usui96


def build_BC(use_benison: bool = True):
    """Build a single bipolar cell with the Usui 1996 channels.

    Can switch to using the calcium dynamics from RGCs (Benison et al., 2001) because
    the pump is less computationally expensive.
    """
    comp = jx.Compartment()
    branch = jx.Branch(comp, ncomp=1)
    BC = jx.Cell([branch], [-1])

    BC.compute_xyz()

    # Insert the mechanisms
    BC.insert(usui96.Leak(name="BC_Leak"))
    BC.insert(usui96.Kv(name="BC_Kv"))
    BC.insert(usui96.KA(name="BC_KA"))
    BC.insert(usui96.Hyper(name="BC_Hyper", solver="explicit"))

    # Insert the calcium mechanisms
    if use_benison:
        BC.insert(benison01.CaL(name="BC_CaL"))
        BC.insert(benison01.CaN(name="BC_CaN"))
        BC.insert(benison01.CaPumpNS(name="BC_CaPumpNS"))
        BC.insert(benison01.CaNernstReversal(name="BC_CaNernstReversal"))
        BC.insert(benison01.KCa(name="BC_KCa"))
    else:
        BC.insert(usui96.KCa(name="BC_KCa"))
        BC.insert(usui96.Ca(name="BC_Ca"))
        BC.insert(usui96.CaNernstReversal(name="BC_CaNernstReversal"))
        BC.insert(usui96.CaPump(name="BC_CaPump", solver="explicit"))

    BC.init_states()

    return BC
