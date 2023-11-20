import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import corner
import torch
import mplhep as hep
import pickle as pkl

from .transforms import original_ranges

hep.style.use("CMS")


def dump_main_plot(
    arr1,
    arr2,
    var_name,
    nbins,
    range,
    labels,
):
    fig, (up, down) = plt.subplots(
        nrows=2,
        ncols=1,
        gridspec_kw={"height_ratios": (2, 1)},
        sharex=True,
    )

    color1 = "g"
    color2 = "b"
    arr1_hist, edges = np.histogram(arr1, bins=nbins, range=range)
    centers = (edges[1:] + edges[:-1]) / 2
    arr1_hist_norm, _, _ = up.hist(
        arr1, bins=nbins, range=range, density=True, label=labels[0], histtype="step", color=color1
    )
    arr1_err_norm = np.sqrt(arr1_hist) / (np.diff(edges) * len(arr1))
    up.errorbar(
        centers,
        arr1_hist_norm,
        yerr=arr1_err_norm,
        color=color1,
        marker="",
        linestyle="",
    ) 
    arr2_hist, _ = np.histogram(arr2, bins=nbins, range=range)
    arr2_hist_norm, _, _ = up.hist(
        arr2, bins=nbins, range=range, density=True, label=labels[1], histtype="step", color=color2
    )
    arr2_err_norm = np.sqrt(arr2_hist) / (np.diff(edges) * len(arr2))
    up.errorbar(
        centers,
        arr2_hist_norm,
        yerr=arr2_err_norm,
        color=color2,
        marker="",
        linestyle="",
    )

    ratio_hist = arr1_hist_norm / arr2_hist_norm
    ratio_err = np.sqrt((arr1_err_norm / arr1_hist_norm) ** 2 + (arr2_err_norm / arr2_hist_norm) ** 2) * ratio_hist
    down.errorbar(
        centers,
        ratio_hist,
        yerr=ratio_err,
        color="k",
        marker="o",
        linestyle="",
    )    

    # cosmetics
    up.set_ylabel("Normalized yield")
    down.set_ylabel("Ratio")
    down.set_xlabel(var_name)
    up.set_xlim(range[0], range[1])
    down.set_ylim(0.5, 1.5)
    down.axhline(
        1,
        color="grey",
        linestyle="--",
    )
    y_minor_ticks = np.arange(0.5, 1.5, 0.1)
    down.set_yticks(y_minor_ticks, minor=True)
    down.grid(True, alpha=0.4, which="minor")
    up.legend()
    hep.cms.label(
        loc=0, data=True, llabel="Work in Progress", rlabel="", ax=up, pad=0.05
    )

    return fig, (up, down)


def sample_and_plot(
    test_loader,
    model,
    epoch,
    writer,
    comet_logger,
    context_variables,
    target_variables,
    device,
):
    context_size = len(context_variables)
    target_size = len(target_variables)
    with torch.no_grad():
        gen, reco, samples = [], [], []
        for context, target in test_loader:
            context = context.to(device)
            target = target.to(device)
            sample = model.sample(num_samples=1, context=context)
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

    # plot the reco and sampled distributions
    for var in target_variables:
        mn = min(reco[var].min(), samples[var].min())
        mx = max(reco[var].max(), samples[var].max())
        fig, ax = dump_main_plot(
            reco[var],
            samples[var],
            var,
            100,
            (mn, mx),
            ["reco", "sampled"], 
        )
        if device == 0 or type(device) != int:
            fig_name = f"{var}_reco_sampled_transformed.png"
            writer.add_figure(fig_name, fig, epoch)
            comet_logger.log_figure(fig_name, fig, step=epoch)

    # plot after preprocessing back
    preprocess_dct = f"/work/gallim/SIMStudies/FlashSimStudies/preprocessing/preprocessed_photons/pipelines.pkl"
    with open(preprocess_dct, "rb") as f:
        preprocess_dct = pkl.load(f)
    preprocess_dct = preprocess_dct["pipe0"]
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
    reco_back = pd.DataFrame(reco_back)
    samples_back = pd.DataFrame(samples_back)
    for var in target_variables:
        fig, ax = dump_main_plot(
            reco_back[var],
            samples_back[var],
            var,
            100,
            original_ranges[var],
            ["reco", "sampled"], 
        )
        if device == 0 or type(device) != int:
            fig_name = f"{var}_reco_sampled.png"
            writer.add_figure(fig_name, fig, epoch)
            comet_logger.log_figure(fig_name, fig, step=epoch)

    # corner plots with target variables
    fig = corner.corner(
        reco_back[target_variables],
        labels=target_variables,
        range=[original_ranges[var] for var in target_variables],
        quantiles=[0.5, 0.9, 0.99],
        color="g",
        hist_bin_factor=3,
        plot_datapoints=False,
        scale_hist=True,
        label_kwargs=dict(fontsize=10),
    )
    corner.corner(
        samples_back[target_variables],
        labels=target_variables,
        range=[original_ranges[var] for var in target_variables],
        quantiles=[0.5, 0.9, 0.99],
        color="b",
        hist_bin_factor=3,
        plot_datapoints=False,
        scale_hist=True,
        label_kwargs=dict(fontsize=12),
        fig=fig,
    )
    for ax in fig.get_axes():
        ax.tick_params(axis="both", labelsize=12)
    # add legend
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color="g", lw=2),
            plt.Line2D([0], [0], color="b", lw=2),
        ],
        labels=["reco", "sampled"],
        loc="upper right",
        fontsize=28,
    )
    if device == 0 or type(device) != int:
        fig_name = "corner_reco_sampled.png"
        writer.add_figure(fig_name, fig, epoch)
        comet_logger.log_figure(fig_name, fig, step=epoch)