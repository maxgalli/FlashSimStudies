import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import corner
import torch
import mplhep as hep
import pickle as pkl

from .transforms import original_ranges

hep.style.use("CMS")


def divide_dist(distribution, bins):
    sorted_dist = np.sort(distribution)
    subgroup_size = len(distribution) // bins
    edges = [sorted_dist[0]]
    for i in range(subgroup_size, len(sorted_dist), subgroup_size):
        edges.append(sorted_dist[i])
    edges[-1] = sorted_dist[-1]
    return edges


def interpolate_weighted_quantiles(values, weights, quantiles):
    i = np.argsort(values)
    c = np.cumsum(weights[i])
    return values[i[np.searchsorted(c, np.array(quantiles) * c[-1])]]


def dump_profile_plot(
    ax, ss_name, cond_name, sample_name, ss_arr, cond_arr, color, cond_edges, weights
):
    df = pd.DataFrame({ss_name: ss_arr, cond_name: cond_arr, "weights": weights})
    quantiles = [0.25, 0.5, 0.75]
    qlists = [[], [], []]
    centers = []
    for left_edge, right_edge in zip(cond_edges[:-1], cond_edges[1:]):
        dff = df[(df[cond_name] > left_edge) & (df[cond_name] < right_edge)]
        # procedure for weighted quantiles
        data = dff[ss_name].values
        weights = dff["weights"].values
        qlist = interpolate_weighted_quantiles(data, weights, quantiles)
        for i, q in enumerate(qlist):
            qlists[i].append(q)
        centers.append((left_edge + right_edge) / 2)
    mid_index = len(quantiles) // 2
    for qlist in qlists[:mid_index]:
        ax.plot(centers, qlist, color=color, linestyle="dashed")
    for qlist in qlists[mid_index:]:
        ax.plot(centers, qlist, color=color, linestyle="dashed")
    ax.plot(centers, qlists[mid_index], color=color, label=sample_name)

    return ax


def dump_full_profile_plot(
    nbins,
    target_variable,
    cond_variable,
    reco_df,
    sample_df,
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    reco_arr = reco_df[target_variable].values
    reco_cond_arr = reco_df[cond_variable].values
    sample_arr = sample_df[target_variable].values
    sample_cond_arr = sample_df[cond_variable].values
    cond_edges = divide_dist(reco_cond_arr, nbins)

    for name, ss_arr, cond_arr, color, w in [
        ("reco", reco_arr, reco_cond_arr, "blue", np.ones(len(reco_arr))),
        ("sampled", sample_arr, sample_cond_arr, "green", np.ones(len(sample_arr))),
    ]:
        ax = dump_profile_plot(
            ax=ax,
            ss_name=target_variable,
            cond_name=cond_variable,
            sample_name=name,
            ss_arr=ss_arr,
            cond_arr=cond_arr,
            color=color,
            cond_edges=cond_edges,
            weights=w,
        )
    ax.legend()
    ax.set_xlabel(cond_variable)
    ax.set_ylabel(target_variable)
    # reduce dimension of labels and axes names
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["xtick.labelsize"] = 12
    plt.rcParams["ytick.labelsize"] = 12
    plt.rcParams["legend.fontsize"] = 12
    fig.tight_layout()

    return fig, ax


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

    try: 
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
    except ZeroDivisionError:
        print("ZeroDivisionError")
        pass

    # cosmetics
    if var_name in ["RecoPhoGenPho_deltaeta", "RecoPho_sieip", "RecoPho_pfRelIso03_chg", "RecoPho_pfRelIso03_all", "RecoPho_hoe", "RecoPho_esEffSigmaRR"]:
        up.set_yscale("log")
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
        mn = mn if mn > -10 else -5
        mx = max(reco[var].max(), samples[var].max())
        mx = mx if mx < 10 else 5
        fig, ax = dump_main_plot(
            reco[var],
            samples[var],
            var,
            100,
            (mn, mx),
            ["reco", "sampled"], 
        )
        if device == 0 or type(device) != int:
            fig_name = f"{var}_reco_sampled_transformed"
            writer.add_figure(fig_name, fig, epoch)
            comet_logger.log_figure(fig_name, fig, step=epoch)

    # plot after preprocessing back
    preprocess_dct = test_loader.dataset.pipelines
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
            fig_name = f"{var}_reco_sampled"
            writer.add_figure(fig_name, fig, epoch)
            comet_logger.log_figure(fig_name, fig, step=epoch)

    # profile plots
    # attach context variables to the reco and sampled dataframes
    for var in context_variables:
        var_back = (
            preprocess_dct[var]
            .inverse_transform(gen[var].values.reshape(-1, 1))
            .reshape(-1)
        )
        reco_back[var] = var_back
        samples_back[var] = var_back
    for var in target_variables:
        nbins_profile = 8
        for cond_var in context_variables:
            if cond_var not in ["GenPho_status", "PU_pudensity"]:
                print(f"Plotting {var} vs {cond_var}")
                fig, ax = dump_full_profile_plot(
                    nbins_profile,
                    var,
                    cond_var,
                    reco_back,
                    samples_back,
                )
                if device == 0 or type(device) != int:
                    fig_name = f"profile_{var}_vs_{cond_var}"
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
        fig_name = "corner_reco_sampled"
        writer.add_figure(fig_name, fig, epoch)
        comet_logger.log_figure(fig_name, fig, step=epoch)