import jax.numpy as jnp
import numpy as np



def coords_to_gridcoords(coords, im):
    """Map coordinates to image grid coordinates.
    
    coords[0] (x) maps to image columns (second index)
    coords[1] (y) maps to image rows (first index)
    """
    img_height, img_width = im.shape[0], im.shape[1]
    
    # First normalize coordinates to [0, 1] range
    x_normalized = (coords[0] + 2400) / (2 * 2400)
    y_normalized = (coords[1] + 2400) / (2 * 2400)
    
    # Then scale to image dimensions
    grid_x = np.round(x_normalized * (img_width - 1)).astype(int)
    grid_y = np.round(y_normalized * (img_height - 1)).astype(int)
    
    # Since the origin is at the top in images but typically at bottom in plots,
    # flip the y-coordinate
    grid_y = img_height - 1 - grid_y
    
    # Make sure coordinates are in bounds
    grid_y = np.clip(grid_y, 0, img_height-1)
    grid_x = np.clip(grid_x, 0, img_width-1)
    
    return grid_x, grid_y


def stimulus_from_image(
        im: np.ndarray, 
        coords: np.ndarray, 
        stim_vars: dict
        ) -> jnp.ndarray:
    """Turn the images into arrays for data_clamp() in simulate().
    
    NOTE: vmap over this function to create a batch of image stimuli. Not jittable though!
    """
    grid_x, grid_y = coords_to_gridcoords(coords, im)
    
    # Extract intensities using conventional NumPy indexing [row, column] ([y, x] in image/array coordinates)
    intensities = im[grid_y, grid_x, 0] * stim_vars["stim_max"]
    
    # Rest of your function remains the same
    time = jnp.arange(0, stim_vars["t_max"], stim_vars["dt"]) 
    stim_vec = jnp.where((time >= stim_vars["peak_start"]) & (time <= stim_vars["peak_end"]), 1, 0)    
    return jnp.outer(intensities, stim_vec)
