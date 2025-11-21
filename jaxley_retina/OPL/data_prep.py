import h5py
import jaxley_retina
from jaxley_retina.cell_embedding import circular_square_lattice
import numpy as np
import pandas as pd
import pathlib
from scipy.integrate import trapezoid
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def michelson_contrast(AreaBlue, AreaGreen):
    """Compute the spectral constrast with the paper equations."""
    abs_green = abs(AreaGreen)
    abs_blue = abs(AreaBlue)
    if AreaBlue < 0 and AreaGreen < 0:
        SC = (abs_green - abs_blue) / (abs_green + abs_blue)  # equation 6-a
    elif AreaBlue < 0 and AreaGreen > 0:
        SC = -1 - (abs_green / abs_blue)  # equation 6-c
    elif AreaBlue > 0 and AreaGreen < 0:
        SC = 1 + (abs_blue / abs_green)  # equation 6-b
    else:
        SC = np.nan
    return SC


def get_selectivities(traces, stim_type):
    """Estimate the location of a scan along the dorsal-ventral axis by
    computing the spectral constrast value of each ROI."""
    nRois = traces.shape[1]
    ReadOutLength = 600
    fs = 1 / 500

    if stim_type == "ff_glut":
        Baseline_Start = 870
        BaselineLength = 60
        BlueStart = 0
        GreenStart = 1000

        SCs = np.empty(nRois)
        SCs[:] = np.nan

        for r in range(0, nRois):
            # Remove the baseline as done by the paper's code
            curr_trace = traces[:, r]
            Baseline = np.mean(
                curr_trace[Baseline_Start : Baseline_Start + BaselineLength]
            )
            curr_trace -= Baseline
            # Get the areas
            AreaBlue = trapezoid(
                curr_trace[BlueStart : BlueStart + ReadOutLength], dx=fs
            )
            AreaGreen = trapezoid(
                curr_trace[GreenStart : GreenStart + ReadOutLength], dx=fs
            )
            # Estimate the spectral contrast
            SCs[r] = michelson_contrast(AreaBlue, AreaGreen)

    if stim_type == "cs_glut":
        BlueStart = 2000
        GreenStart = 0
        BlueStart_Surr = 3000
        GreenStart_Surr = 1000
        BaselineLength = 200

        SCs = np.empty((nRois, 2))
        SCs[:, :] = np.nan

        for r in range(0, nRois):
            # Remove the baseline as done by the paper's code
            curr_trace = traces[:, r]
            curr_trace -= curr_trace[0]
            # Get the green center trace area
            AreaGreen = trapezoid(
                curr_trace[GreenStart : GreenStart + ReadOutLength], dx=fs
            )

            # Remove the next baseline as done by the paper's code
            Baseline = np.nanmean(curr_trace[BlueStart - BaselineLength : BlueStart])
            # Get the blue center trace area
            AreaBlue = trapezoid(
                curr_trace[BlueStart : BlueStart + ReadOutLength], dx=fs
            )

            # Estimate the center SC
            SCs[r, 0] = michelson_contrast(AreaBlue, AreaGreen)

            # Get the surround trace areas
            Baseline = np.nanmean(
                curr_trace[BlueStart_Surr - BaselineLength : BlueStart_Surr]
            )
            curr_trace -= Baseline
            AreaBlue_Surr = trapezoid(
                curr_trace[BlueStart_Surr : BlueStart_Surr + ReadOutLength], dx=fs
            )
            Baseline = np.nanmean(
                curr_trace[GreenStart_Surr - BaselineLength : GreenStart_Surr]
            )
            curr_trace -= Baseline
            AreaGreen_Surr = trapezoid(
                curr_trace[GreenStart_Surr : GreenStart_Surr + ReadOutLength], dx=fs
            )

            # NOTE: redundant filtering in paper code here omitted (if there
            # isn't a depolarizing surround response, then none of the conds
            # will be met)

            # Get dominant center area (does green or blue depolarize more)
            center_area = abs(min(AreaGreen, AreaBlue))

            # Estimate SC if surround area is at least 10% of the center area
            thresh = 0.1
            conds = np.array(
                [
                    AreaBlue_Surr / center_area,
                    AreaGreen_Surr / center_area,
                    (AreaBlue_Surr + AreaGreen_Surr) / center_area,
                ]
            )
            if (conds > thresh).any():
                SCs[r, 1] = michelson_contrast(AreaBlue, AreaGreen)

    # Set SC to zero where nan
    SCs = np.nan_to_num(SCs, copy=True, nan=0.0)
    # Deal with the edges of the SC estimates
    SCs = np.clip(SCs, -1, 1)

    return SCs


def sort_rois(roi_df):
    """Get the indices for each of the scans"""
    idx_sets = roi_df.groupby(["Date", "L/R", "Scan Field"]).indices
    idx_sets = list(idx_sets.values())
    return idx_sets

# Global I/O path for the following functions
p = pathlib.Path(__file__).parent.resolve().parents[1] / "data"

def get_opsin_densities():
    # Load the experimental data about selectivites
    nadal_data_path = p / "Nadal2020_Fig3c_PigGrads.csv"
    opsin_data = pd.read_csv(nadal_data_path)

    # Build a large grid over which to interpolate opsin selectivities
    coords = circular_square_lattice(30_000, ret_rad=2400)

    # Define the interpolation function
    def _euc_interp(coords, x, y, z):
        # Find the two closest coordinates
        dists = np.linalg.norm(coords - np.stack((x, y), axis=1), axis=1)
        idxs = np.argsort(dists)[:2]
        # Interpolate between the z values at those coordinates
        dist_weights = dists[idxs] / np.sum(dists[idxs])
        z1 = z[idxs[0]]
        z2 = z[idxs[1]]
        return np.average([z1, z2], weights=dist_weights)

    # Rotate the coords to match the opsin_data coords
    rot_mat = np.array(
        [
            [np.cos(np.pi / 4), -np.sin(np.pi / 4)],
            [np.sin(np.pi / 4), np.cos(np.pi / 4)],
        ]
    )
    rot_coords = np.matmul(rot_mat, coords)

    # Map the function over all the different cone combos and coords
    mixed_d = np.apply_along_axis(
        _euc_interp,
        0,
        rot_coords,
        opsin_data["DT-VN (um)"],
        opsin_data["VT-DN (um)"],
        opsin_data["Mixed Avg"],
    )
    truem_d = np.apply_along_axis(
        _euc_interp,
        0,
        rot_coords,
        opsin_data["DT-VN (um)"],
        opsin_data["VT-DN (um)"],
        opsin_data["True M Avg"],
    )
    trues_d = np.apply_along_axis(
        _euc_interp,
        0,
        rot_coords,
        opsin_data["DT-VN (um)"],
        opsin_data["VT-DN (um)"],
        opsin_data["True S Avg"],
    )

    all_d = np.vstack((mixed_d, truem_d, trues_d))
    all_d = all_d / np.sum(all_d, axis=0)
    return coords, all_d


def estimate_scan_loc(SCs, all_d, coords):
    """Divide the SCs given by their selectivity."""
    SC_uv = np.where(SCs == -1)[0]
    SC_mixed = np.where(abs(SCs) != 1)[0]
    SC_g = np.where(SCs == 1)[0]
    SC_densities = np.array([[len(SC_mixed), len(SC_g), len(SC_uv)]]) / len(SCs)

    dists = cdist(all_d.T, SC_densities)
    _, all_d_inds = linear_sum_assignment(dists.T)
    x, y = coords[:, all_d_inds[0]]
    return (x, y)


def organize_data():
    """
    Organize the Zenodo data: filter out the nan trials and low quality trials, and
    divide traces by scan location.
    """
    data_load_path = p / "Data_Cones.h5"
    data_save_path = p / "cone_data_uncentered.h5"
    f_in = h5py.File(data_load_path, "r")
    # Load glutamate traces (full-field and center-surround)
    ff_glut = np.array(f_in["BGW_Traces"])
    cs_glut = np.array(f_in["BG_CS_Traces"])

    # Load the trace quality indices
    ff_qual = np.array(f_in["BGW_Quality"])
    cs_qual = np.array(f_in["BG_CS_Quality"])

    # Load ROI information
    descripts = [
        "Date",
        "L/R",
        "Scan Field",
        "ROI",
        "x-coord",
        "y-coord",
        "Zoom Factor",
    ]
    roi_df = pd.DataFrame(data=f_in["RoiInfo"], columns=descripts)

    # Filter out ROI indices that have problems for either stimulus
    lowq_ff = np.where(ff_qual < 0.25)[0]
    lowq_cs = np.where(cs_qual < 0.25)[0]
    # ROIs at the left of the scan field (20 pixels)
    left_rois = np.where(roi_df["x-coord"] < 20)[0]
    # Nan trial ROI indices
    nan_cols = np.sum(np.isnan(ff_glut), axis=0)
    nan_trials_ff = np.where(nan_cols == ff_glut.shape[0])[0]
    nan_cols = np.sum(np.isnan(cs_glut), axis=0)
    nan_trials_cs = np.where(nan_cols == cs_glut.shape[0])[0]
    cut_inds = list(
        set(lowq_ff)
        | set(lowq_cs)
        | set(left_rois)
        | set(nan_trials_ff)
        | set(nan_trials_cs)
    )

    # Cut these indices out of the roi_df
    roi_df.drop(cut_inds, inplace=True)
    ff_glut = np.delete(ff_glut, cut_inds, axis=1)
    cs_glut = np.delete(cs_glut, cut_inds, axis=1)

    # Get indices of ROIs in the same scan field
    roi_sets = sort_rois(roi_df)

    # Set up the dataset
    f_out = h5py.File(data_save_path, "w")
    [f_out.create_group(str(i)) for i in range(len(roi_sets))]

    # Get the opsin densities for estimating the scan locations later
    coords, all_densities = get_opsin_densities()

    for i, s in zip(np.arange(len(roi_sets), dtype=int).astype(str), roi_sets):

        # Only continue if there are ROIs with data
        if len(s) > 0:

            # Save these glutamate traces in the data set
            f_out[i]["ff_glut"] = ff_glut[:, s]
            f_out[i]["cs_glut"] = cs_glut[:, s]

            # Calculate the spectral selectivities based on the ff data
            SCs = get_selectivities(ff_glut[:, s], "ff_glut")
            f_out[i]["SC"] = SCs

            # Only save the scan field for the ff stimulus as before
            scan_loc = estimate_scan_loc(SCs, all_densities, coords)
            f_out[i]["scan_loc"] = scan_loc

            # Convert the ROI locations in pixel space to um and center
            pix_coords = np.array(
                [roi_df["x-coord"].to_numpy()[s], roi_df["y-coord"].to_numpy()[s]]
            ).T
            um_coords = pix_coords * (110 / 128) - (110 / 2)
            # Add scan_loc estimate to the centered coordinates
            um_coords[:, 0] += f_out[i]["scan_loc"][0]
            um_coords[:, 1] += f_out[i]["scan_loc"][1]
            f_out[i]["roi_locs"] = um_coords


def choose_traces():
    """Load cone_data_uncentered.h5 (all scan fields organized) and choose traces"""
    data_load_path = p / "cone_data_uncentered.h5"
    data = h5py.File(data_load_path, "r")

    # Gather all traces and divide into ventral and dorsal traces
    all_cs_traces = [np.array(data[str(i)]["cs_glut"]).T for i in range(len(data))]
    sf_locs = np.array([data[str(i)]["scan_loc"][1] for i in range(len(data))])
    dorsal_traces = np.vstack([all_cs_traces[i] for i in np.where(sf_locs > 0)[0]])
    ventral_traces = np.vstack([all_cs_traces[i] for i in np.where(sf_locs < 0)[0]])
    fs = 1 / 500

    # Filter the data
    inds = []
    for traces in (dorsal_traces, ventral_traces):
        segments = [traces[:, int(i / fs) : int((i + 2) / fs)] for i in range(0, 8, 2)]
        means = [np.mean(segment, axis=1) for segment in segments]

        green_surround_pos = np.where(means[1] > 0.25)[0]
        green_center_neg = np.where(means[0] < 0.0)[0]
        uv_center_neg = np.where(means[2] < 0.0)[0]

        center_neg = np.intersect1d(green_center_neg, uv_center_neg)
        inds.append(np.intersect1d(green_surround_pos, center_neg))

    # 60 here because 63 is max number of dorsal qualifying traces
    trace_collection = np.concatenate(
        (dorsal_traces[inds[0][:60]], ventral_traces[inds[1][:60]]), axis=0
    )
    # Set the beginning of the trace to zero
    trace_collection = np.subtract(trace_collection.T, trace_collection[:, 0]).T
    data_save_path = p / "120_center_responses.npy"
    np.save(str(data_save_path), trace_collection, allow_pickle=False)


if __name__ == "__main__":
    organize_data()
    choose_traces()
