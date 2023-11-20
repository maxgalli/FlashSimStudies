import hydra
from hydra.utils import get_original_cwd
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import logging
import os
import sys
import time
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import pickle as pkl

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.profiler import profile, record_function, ProfilerActivity

# where wipfs modules are
sys.path.append("/work/gallim/SIMStudies/wipfs/models")
sys.path.append("/work/gallim/SIMStudies/FlashSimStudies/preprocessing")

# seed
np.random.seed(42)

from modded_basic_nflow import create_mixture_flow_model, load_mixture_model, save_model
from preprocess_photons import original_ranges
from train_photons import PhotonDataset

def plot_histograms(reco, samples, target_variables, output_dir, suffix=""):
    for var in target_variables:
        # remove inf and -inf from samples
        samples[var] = samples[var][np.isfinite(samples[var])]
        mn = min(reco[var].min(), samples[var].min())
        mx = max(reco[var].max(), samples[var].max())
        fig, ax = plt.subplots(1, 1, figsize=(15, 10), tight_layout=True)
        ax.hist(reco[var], bins=100, histtype="step", label="reco", range=(mn, mx), density=True)
        ws = wasserstein_distance(reco[var], samples[var])
        ax.hist(
            samples[var],
            bins=100,
            histtype="step",
            label=f"sampled (wasserstein={ws:.3f})",
            range=(mn, mx),
            density=True,
        )
        ax.set_xlabel(var)
        ax.legend()
        for ext in ["png", "pdf"]:
            fig.savefig(f"{output_dir}/{var}{suffix}.{ext}")
   

def apply_model(cfg):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # create dict as required by the function
    input_dim = len(cfg.target_variables)
    context_dim = len(cfg.context_variables)
    flow_params_dct = {
        "input_dim": input_dim,
        "context_dim": context_dim,
        "base_kwargs": {
            "num_steps_maf": cfg.model.maf.num_steps,
            "num_steps_arqs": cfg.model.arqs.num_steps,
            "num_transform_blocks_maf": cfg.model.maf.num_transform_blocks,
            "num_transform_blocks_arqs": cfg.model.arqs.num_transform_blocks,
            "activation": cfg.model.activation,
            "dropout_probability_maf": cfg.model.maf.dropout_probability,
            "dropout_probability_arqs": cfg.model.arqs.dropout_probability,
            "use_residual_blocks_maf": cfg.model.maf.use_residual_blocks,
            "use_residual_blocks_arqs": cfg.model.arqs.use_residual_blocks,
            "batch_norm_maf": cfg.model.maf.batch_norm,
            "batch_norm_arqs": cfg.model.arqs.batch_norm,
            "num_bins_arqs": cfg.model.arqs.num_bins,
            "tail_bound_arqs": cfg.model.arqs.tail_bound,
            "hidden_dim_maf": cfg.model.maf.hidden_dim,
            "hidden_dim_arqs": cfg.model.arqs.hidden_dim,
            "init_identity": cfg.model.init_identity,
        },
        "transform_type": cfg.model.transform_type,
    }
    model = create_mixture_flow_model(**flow_params_dct)
    model, _, _, start_epoch, _, _ = load_mixture_model(
        model, model_dir=cfg.checkpoint, filename="checkpoint-latest.pt"
    )
    model = model.to(device)

    # make datasets
    print("Loading datasets...")
    test_dataset = PhotonDataset(
        cfg.test.path,
        cfg.context_variables,
        cfg.target_variables,
        device=device,
        #rows=10000
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        # num_workers=2,
        #pin_memory=True,
    )

    # sample
    output_dir = "/work/gallim/SIMStudies/FlashSimStudies/training/ForMattia"
    context_variables = cfg.context_variables
    target_variables = cfg.target_variables
    target_size = len(target_variables)
    print("Sampling...")
    with torch.no_grad():
        gen, reco, samples = [], [], []
        for context, target in test_loader:
            context = context.to(device)
            target = target.to(device)
            sample = model.sample(num_samples=10, context=context)
            context = context.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            sample = sample.detach().cpu().numpy()
            sample = sample.reshape(-1, target_size)
            gen.append(context)
            reco.append(target)
            samples.append(sample)
    gen = np.concatenate(gen, axis=0)
    reco = np.concatenate(reco, axis=0)
    samples = np.concatenate(samples, axis=0)
    gen = pd.DataFrame(gen, columns=context_variables)
    reco = pd.DataFrame(reco, columns=target_variables)
    samples = pd.DataFrame(samples, columns=target_variables)
    print("Done sampling")

    preprocess_dct = f"/work/gallim/SIMStudies/FlashSimStudies/preprocessing/preprocessed_photons/pipelines.pkl"
    with open(preprocess_dct, "rb") as f:
        preprocess_dct = pkl.load(f)
    gen_back = {}
    reco_back = {}
    samples_back = {}
    for var in target_variables:
        reco_back[var] = (
            preprocess_dct[var]
            .inverse_transform(reco[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        samples_back[var] = (
            preprocess_dct[var]
            .inverse_transform(samples[var].values.reshape(-1, 1))
            .reshape(-1)
        )
    for var in context_variables:
        gen_back[var] = (
            preprocess_dct[var]
            .inverse_transform(gen[var].values.reshape(-1, 1))
            .reshape(-1)
        )
    gen_back = pd.DataFrame(gen_back)
    reco_back = pd.DataFrame(reco_back)
    samples_back = pd.DataFrame(samples_back)

    # save to parquet
    for name, df in zip(["gen", "reco", "samples", "gen_back", "reco_back", "samples_back"], [gen, reco, samples, gen_back, reco_back, samples_back]):
        df.to_parquet(os.path.join(output_dir, f"{name}.parquet"))

    # plot
    print("Plotting and saving...")
    plot_histograms(
        reco, samples, target_variables=target_variables, output_dir=output_dir
    )
    plot_histograms(
        reco_back, samples_back, target_variables=target_variables, output_dir=output_dir, suffix="_back"
    )

@hydra.main(version_base=None, config_path="config_photons", config_name="cfg0")
def main(cfg):
    # This because in the hydra config we enable the feature that changes the cwd to the experiment dir
    initial_dir = get_original_cwd()
    print("Initial dir: ", initial_dir)
    print("Current dir: ", os.getcwd())
    apply_model(cfg)


if __name__ == "__main__":
    main()