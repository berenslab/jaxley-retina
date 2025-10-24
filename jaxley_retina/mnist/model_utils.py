import jaxley as jx
from jaxley_mech.channels.usui96 import Leak
import numpy as np

from jaxley_retina.cell_embedding import circular_square_lattice
from jaxley_retina.IPL.BC import build_BC


def build_readout(BC_readouts=False, use_benison=True):
    """Builds either a BC readout or a single compartment readout with a leak channel."""
    if BC_readouts:
        readout = build_BC(use_benison=use_benison)
    else:
        comp = jx.Compartment()
        branch = jx.Branch(comp, ncomp=1)
        readout = jx.Cell([branch], [-1])
        readout.compute_xyz()
        readout.insert(Leak())
    return readout

def get_coords(n: int, rad=2400) -> np.ndarray:
    grid = circular_square_lattice(n, ret_rad=rad)
    grid = np.vstack((grid, np.zeros(n))) # add z=0
    return grid