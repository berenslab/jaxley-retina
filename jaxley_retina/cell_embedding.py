import numpy as np


def circular_square_lattice(num_cells: int, ret_rad: int = 2400) -> np.ndarray:
    """
    Make circle-shaped lattice with exactly the number of cells specified.
    The function generates a grid of points within a circle of radius ret_rad,
    and then selects exactly num_cells points from this grid.

    Parameters:
    -----------
    num_cells : int
        The exact number of cells to include in the lattice
    ret_rad : int, optional
        The radius of the circle in micrometers, default is 2400

    Returns:
    --------
    np.ndarray
        Array of shape (2, num_cells) containing the x,y coordinates
    """
    grid_res = np.ceil(ret_rad / np.sqrt(num_cells / np.pi)).astype(int)

    # Make the square grid
    xcoords = np.tile(
        np.arange(-ret_rad, ret_rad, step=grid_res),
        ((np.ceil(ret_rad * 2 / grid_res).astype(int), 1)),
    )
    ycoords = xcoords.copy().T
    square_coords = np.stack((xcoords, ycoords), axis=0)

    # Find all the coordinates within the circle contained
    dists = np.linalg.norm(square_coords, axis=0)
    circ_idxs = np.where(dists <= ret_rad)
    circ_coords = square_coords[:, circ_idxs[0], circ_idxs[1]]

    # Get the actual number of cells we generated
    actual_cells = circ_coords.shape[1]

    # If we have too many cells, select nearest exactly num_cells points
    if actual_cells > num_cells:
        dists = np.linalg.norm(circ_coords, axis=0)
        selected_indices = np.argsort(dists)[:num_cells]
        circ_coords = circ_coords[:, selected_indices]

    # If we have too few cells, decrease the grid resolution and try again
    elif actual_cells < num_cells:
        while actual_cells < num_cells:
            # Try a smaller grid resolution
            grid_res -= int(np.sqrt(num_cells))
            # TODO: find a nicer way to handle this case
            if grid_res <= 0:
                # find a resolution between 0 and 1
                frac = (num_cells - actual_cells) / num_cells
                grid_res = (1 - frac) * int(np.sqrt(num_cells))

            # Regenerate the grid with new resolution
            xcoords = np.tile(
                np.arange(-ret_rad, ret_rad, step=grid_res),
                ((np.ceil(ret_rad * 2 / grid_res).astype(int), 1)),
            )
            ycoords = xcoords.copy().T
            square_coords = np.stack((xcoords, ycoords), axis=0)

            # Recalculate circle coordinates
            circ_idxs = np.where(np.linalg.norm(square_coords, axis=0) <= ret_rad)
            circ_coords = square_coords[:, circ_idxs[0], circ_idxs[1]]

            actual_cells = circ_coords.shape[1]

        # If we no have too many cells (as opposed to the exact number)
        if actual_cells > num_cells:
            dists = np.linalg.norm(circ_coords, axis=0)
            selected_indices = np.argsort(dists)[:num_cells]
            circ_coords = circ_coords[:, selected_indices]

    return circ_coords
