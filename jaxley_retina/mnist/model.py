import jaxley as jx
from jaxley.synapses import IonotropicSynapse
from jaxley_mech.synapses import RibbonSynapse
from jaxley_retina.OPL.PR import build_PR
from jaxley_retina.OPL.HC import build_HC
from jaxley_retina.mnist.train_io import load_PTC_params, load_ribbon_params
from jaxley_retina.mnist.model_utils import build_readout, get_coords
from jaxley_retina.mnist.transforms import mnist_transform

import numpy as np
from scipy.spatial import distance_matrix



def build_mnist_model(model_config: dict) -> jx.Network:
    # Load PTC params
    ptc_path = model_config['ptc_path']
    ptc_params = load_PTC_params(ptc_path)

    PR, _ = build_PR(ptc_params) # trained PTC params
    HC = build_HC() 
    readout = build_readout(BC_readouts=model_config['BC_readouts'])

    n_readouts = model_config["n_readouts"]
    nHCs = model_config["nHCs"]
    nPRs = model_config["nPRs"]

    # Build network
    if nHCs == 0:
        network = jx.Network([PR] * nPRs + [readout] * 10)
    else:
        network = jx.Network([HC] * nHCs + [PR] * nPRs + [readout] * n_readouts)
        network[:nHCs].add_to_group('HC')
        HC_inds = network.HC.nodes.global_cell_index.tolist()
        
    network[nHCs:nPRs+nHCs].add_to_group('PR')
    PR_inds = network.PR.nodes.global_cell_index.tolist() 
    network[-n_readouts:].add_to_group('readout') 
    readout_inds = network.readout.nodes.global_cell_index.tolist()

    # Embed the cells in 2D space (not caring about the readouts for now, HCs only for vis)
    grid = get_coords(nPRs)
    network.PR.move_to(grid[0], grid[1], grid[2])
    if nHCs == 1:
        network.HC.move_to(0., 0., 0.)
    elif nHCs > 1:
        grid = get_coords(nHCs)
        network.HC.move_to(grid[0], grid[1], grid[2])

    # Connect the network TODO: improve
    if model_config["connections"] == "full":
        # Connections between each PR and the HC
        jx.fully_connect(network.cell(PR_inds), network.cell(HC_inds), RibbonSynapse(name='RibbonHC', solver='explicit'))
        jx.fully_connect(network.cell(HC_inds), network.cell(PR_inds), IonotropicSynapse())
    elif model_config["connections"] == "one-to-one":
        # Each PR reciprocally connected to one HC
        jx.connectivity_matrix_connect(
            network.cell(PR_inds), 
            network.cell(HC_inds), 
            RibbonSynapse(name='RibbonHC', solver='explicit'),
            np.eye(len(PR_inds), len(HC_inds), dtype=bool)
            )
        jx.connectivity_matrix_connect(
            network.cell(HC_inds),
            network.cell(PR_inds),
            IonotropicSynapse(),
            np.eye(len(HC_inds), len(PR_inds), dtype=bool)
        )
    elif model_config["connections"] == "local":
        n_rad = model_config["n_rad"] # Number of PRs to connect to each HC
        # Calculate the adj mat with the coords and n_rad
        network.HC.compute_compartment_centers()
        HC_coords = np.vstack((network.HC.nodes.x, network.HC.nodes.y)) # 2 x N
        network.PR.compute_compartment_centers()
        PR_coords = np.vstack((network.PR.nodes.x, network.PR.nodes.y)) # 2 x M
        dist_mat = distance_matrix(PR_coords.T, HC_coords.T) # -> N x M
        dists_filtered = np.argsort(dist_mat, axis=0)[:n_rad, :] # -> n_rad x M
        dists_filtered_cols = np.repeat(np.arange(dists_filtered.shape[1]), n_rad)
        adj_mat = np.zeros_like(dist_mat).astype(bool) # N x M
        adj_mat[dists_filtered.flatten(order="F"), dists_filtered_cols] = True
        # Connect the views
        jx.connectivity_matrix_connect(
            network.cell(PR_inds),
            network.cell(HC_inds),
            RibbonSynapse(name='RibbonHC', solver="explicit"),
            adj_mat
        )
        jx.connectivity_matrix_connect(
            network.cell(HC_inds),
            network.cell(PR_inds),
            IonotropicSynapse(),
            adj_mat.T
        )
    jx.fully_connect(network.cell(PR_inds), network.cell(readout_inds), RibbonSynapse(name='RibbonReadout', solver='explicit'))

    # Load some fitted ribbon synapse parameters
    ribbon_params = load_ribbon_params(model_config['ribbon_path'])
    # Randomly sample the parameter sets
    param_assignments = np.random.choice(ribbon_params, model_config['nPRs'], replace=True)

    # Set the ribbon synapse params TODO: work on setting with arrays in Jaxley
    alphas = []
    network.copy_node_property_to_edges('global_cell_index')
    edges_df = network.edges
    for i, c in enumerate(PR_inds):
        for name, param in param_assignments[i].items():
            # Separate out the alphas
            if name == 'alphas':
                alphas.append(float(param[0]))
            else:
                # Find the edges with this presynaptic index (should be 11)
                cell_edges = edges_df.query("pre_global_cell_index == @c")
                edges_by_precell = network.select(edges=cell_edges.index)
                # set the parameters at both the synapses with HCs and with readouts
                if model_config["nHCs"] > 0:
                    nameHC = name.replace("RibbonSynapse", "RibbonHC")
                    edges_by_precell.set(nameHC, float(param[0]))
                namereadout = name.replace("RibbonSynapse", "RibbonReadout")
                edges_by_precell.set(namereadout, float(param[0]))
        
    network.init_states()

    return network


def setup_param_training(network: jx.Network, train_config: dict) -> tuple:
    """Set up the parameter training for the model described above. """
    _ = np.random.seed(train_config["seed"]) # keeping here for reproducability

    # Set the cell types as one column in edges for selecting synapses
    for group in network.group_names:
        inds = network.nodes[network.nodes[group]]['global_cell_index'].to_list()
        network.nodes.loc[inds, "cell_type"] = group
    network.copy_node_property_to_edges("cell_type")

    df = network.edges
    ribbon_to_readout = df.query("pre_cell_type == 'PR' & post_cell_type == 'readout'")
    ribbon2readout_subnetwork = network.select(edges=ribbon_to_readout.index)
    ribbon2readout_subnetwork.edges["controlled_by_param"] = ribbon_to_readout.index
    _ = np.random.seed(train_config["seed"]) # legacy, for reproducability
    gS_init = np.random.normal(
        train_config["RibbonReadout_gS_init_mean"], 
        train_config["RibbonReadout_gS_init_std"], 
        size=ribbon2readout_subnetwork.edges["controlled_by_param"].unique().shape[0]
        ).tolist()
    ribbon2readout_subnetwork.make_trainable("RibbonReadout_gS", gS_init)

    # If RibbonHC is in the model, the HC->PR ionotropic synapse also has to be currently
    if "RibbonHC_gS_init" in train_config:
        ribbon_to_HC = df.query("pre_cell_type == 'PR' & post_cell_type == 'HC'")
        ribbon2HC_subnetwork = network.select(edges=ribbon_to_HC.index)
        ribbon2HC_subnetwork.edges["controlled_by_param"] = ribbon_to_HC.index
        _ = np.random.seed(train_config["seed"])
        gS_init = np.random.normal(
            train_config["RibbonHC_gS_init_mean"], 
            train_config["RibbonHC_gS_init_std"], 
            size=ribbon2HC_subnetwork.edges["controlled_by_param"].unique().shape[0]
            ).tolist()
        ribbon2HC_subnetwork.make_trainable("RibbonHC_gS", gS_init)

        ionotropic_df = df.query("pre_cell_type == 'HC'")
        subnetwork = network.select(edges=ionotropic_df.index)
        subnetwork.edges["controlled_by_param"] = ionotropic_df.index
        _ = np.random.seed(train_config["seed"]) # for reproducability
        gS_init = np.random.normal(
            train_config["IonotropicSynapse_gS_init_mean"], 
            train_config["IonotropicSynapse_gS_init_std"], 
            size=subnetwork.edges["controlled_by_param"].unique().shape[0]
        ).tolist()
        subnetwork.make_trainable("IonotropicSynapse_gS", gS_init)

    network.nodes.drop(columns=["cell_type"], inplace=True)

    # Define the parameter transform
    bounds = {"RibbonReadout_gS": (train_config["RibbonReadout_gS_lower"], train_config["RibbonReadout_gS_upper"])}
    if "IonotropicSynapse_gS_init" in train_config:
        bounds.update({
            "RibbonHC_gS": (train_config["RibbonHC_gS_lower"], train_config["RibbonHC_gS_upper"]),
            "IonotropicSynapse_gS": (train_config["IonotropicSynapse_gS_lower"], train_config["IonotropicSynapse_gS_upper"])
        })
    transform = mnist_transform(bounds)

    return network, transform