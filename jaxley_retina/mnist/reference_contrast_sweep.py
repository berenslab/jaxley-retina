from jax import config
config.update("jax_enable_x64", True)
config.update("jax_platform_name", "gpu")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".3"

import argparse
from jax import vmap
from jaxley_retina.mnist.simple_data_prep import build_train_loader
from jaxley_retina.mnist.model import get_coords
from jaxley_retina.mnist.image_to_stim import stimulus_from_image
from jaxley_retina.mnist.reference_models import (
    build_logistic_regression, 
    build_2layer_mlp, 
    build_thresholding_logistic, 
    build_normalizing_logistic
)
import numpy as np
from pathlib import Path
import pickle
import tensorflow as tf
from tqdm import tqdm
import yaml


stimuli_from_imgs = vmap(stimulus_from_image, in_axes=(0, None, None))

def contrast_sweep(contrast_config):

    tf.random.set_seed(contrast_config["seed"])
    _ = np.random.seed(contrast_config["seed"])

    with open(contrast_config["biophysical_path"], "r") as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)

    model_config, stim_config, train_config = (configs["Model config"], configs["Stim config"], configs["Train config"])
    print(stim_config)

    # older configs use phi_max instead of stim_max
    stim_config["stim_max"] = stim_config["phi_max"]
    
    # Have to build the network to get the coords, or be smarter about it, but only have to do once
    coords = get_coords(model_config["nPRs"])[:2, :]  # Only need x and y

    # Build the model
    num_classes = 10 if stim_config["digits"] == "all" else len(stim_config["digits"])
    if contrast_config["model"] == "logistic":
        model = build_logistic_regression((coords.shape[1],), num_classes=num_classes)
    elif contrast_config["model"] == "mlp":
        model = build_2layer_mlp((coords.shape[1],), num_classes=num_classes)
    elif contrast_config["model"] == "static_thresholding_logistic":
        model = build_thresholding_logistic((coords.shape[1],), num_classes=num_classes, adaptive=False)
    elif contrast_config["model"] == "adaptive_thresholding_logistic":
        model = build_thresholding_logistic((coords.shape[1],), num_classes=num_classes, adaptive=True)
    elif contrast_config["model"] == "normalizing_logistic":
        model = build_normalizing_logistic((coords.shape[1],), num_classes=num_classes)
    print(model.summary())

    # Training to convergence
    train_losses = []
    train_accuracies = []

    timept_ms = (stim_config['peak_start']+stim_config['peak_end'])/2
    timept_ts = int(timept_ms / stim_config['dt'])

    epoch_counter = 0
    train_loss_avg = np.inf # intialize loss min
    stop = False

    # Training
    while not stop:
        epoch_counter += 1
        train_losses = []
        
        # Reset data loaders for each epoch
        train_loader, = build_train_loader(
            batch_size=train_config["batch_size"],
            digits=stim_config["digits"],
            contrast_range=(stim_config["train_contrast_min"], stim_config["train_contrast_max"]),
            lum_range=(stim_config["train_lum_min"], stim_config["train_lum_max"]),
            splits=["train"]
        )

        for batch in train_loader:
            x_batch, y_batch = batch
            x_batch = stimuli_from_imgs(x_batch, coords, stim_config)
            x_batch = x_batch[:, :, timept_ts]
            metrics = model.train_on_batch(x_batch, y_batch, return_dict=True)
            train_losses.append(metrics['loss'])
            train_accuracies.append(metrics['accuracy'])

        stop = np.mean(train_losses) >  train_config["stop_criteria"] * train_loss_avg
        # Reset the test_loss_avg
        train_loss_avg = np.mean(train_losses)
        print(f"Epoch: {epoch_counter}, Avg train loss: {train_loss_avg}")

    # Print the threshold if training that
    if contrast_config["model"] == "static_thresholding_logistic" or contrast_config["model"] == "adaptive_thresholding_logistic":
        print(f"Learned threshold: {model.layers[1].threshold.numpy()}")

    # Test in the sweep
    levels = np.flip(np.round(np.arange(0.0, 1.01, step=0.1), 2))
    all_test_accs = {minc: [] for minc in levels}

    for l in levels:
        for _ in tqdm(range(contrast_config["reps"]), desc="Reps"):
            test_accuracies = []
            test_losses = []
                
            if contrast_config["random"]:
                m = stim_config["test_contrast_max"]
            else:
                m = l
            test_loader, = build_train_loader(
                batch_size=train_config["batch_size"],
                digits=stim_config["digits"],
                contrast_range=(l, m),
                lum_range=(stim_config["test_lum_min"], stim_config["test_lum_max"]),
                splits=["test"]
            )
            for batch in test_loader:
                x_batch, y_batch = batch
                x_batch = stimuli_from_imgs(x_batch, coords, stim_config)
                x_batch = x_batch[:, :, timept_ts]
                metrics = model.test_on_batch(x_batch, y_batch, return_dict=True)
                test_losses.append(metrics['loss'])
                test_accuracies.append(metrics['accuracy'])
            
            avg_test_accuracy = np.mean(test_accuracies)
            print(avg_test_accuracy)
            all_test_accs[l].append(avg_test_accuracy)    
        
        print(f"Level: {l}, Test acc: {np.mean(all_test_accs[l])}")

    # Save the test accuracies
    save_path = Path(contrast_config["biophysical_path"])
    save_path = save_path.parents[0] / f"{contrast_config['model']}_contrast_sweep_{contrast_config['seed']}.pkl"
    with open(str(save_path), "wb") as f:
        pickle.dump((contrast_config, all_test_accs), f)

def parse_args():
    """Collect the train config file"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--biophysical_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    contrast_config = {
        "model": args.model,
        "reps": 1,
        "biophysical_path": args.biophysical_path,
        "seed": args.seed,
        "random": False
    }
    contrast_sweep(contrast_config)
