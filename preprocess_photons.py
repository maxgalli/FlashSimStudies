import os
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
import dask.dataframe as dd
import dask
import distributed
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils.transforms import original_ranges, Smearer, Displacer


pipelines = {
    "pipe0": {
        "GenPho_pt": Pipeline(
            [
                (
                    "log_trans",
                    FunctionTransformer(
                        lambda x: np.log1p(x * 0.01),
                        inverse_func=lambda x: np.expm1(x) / 0.01,
                    ),
                ),
                ("scaler", MinMaxScaler((0, 1))),
            ]
        ),
        "GenPho_eta": Pipeline([("scaler", MinMaxScaler((-1, 1)))]),
        "GenPho_phi": Pipeline([("scaler", MinMaxScaler((-1, 1)))]),
        "GenPho_status": Pipeline(
            [("smearer", Smearer("uniform")), ("scaler", MinMaxScaler((-1, 1)))]
        ),
        "GenPhoGenEle_deltar": Pipeline([("scaler", MinMaxScaler((-1, 1)))]),
        "ClosestGenJet_pt": Pipeline(
            [
                (
                    "log_trans",
                    FunctionTransformer(
                        lambda x: np.log1p(x * 0.01),
                        inverse_func=lambda x: np.expm1(x) / 0.01,
                    ),
                ),
                ("scaler", MinMaxScaler((0, 1))),
            ]
        ),
        "ClosestGenJet_mass": Pipeline([("scaler", MinMaxScaler((-1, 1)))]),
        "PU_gpudensity": Pipeline([("none", None)]),
        "PU_nPU": Pipeline([("scaler", MinMaxScaler((-1, 1)))]),
        "PU_nTrueInt": Pipeline([("scaler", MinMaxScaler((-1, 1)))]),
        "PU_pudensity": Pipeline(
            [("smearer", Smearer("uniform")), ("scaler", MinMaxScaler((0, 1)))]
        ),
        "PU_sumEOOT": Pipeline([("scaler", MinMaxScaler((-1, 1)))]),
        "PU_sumLOOT": Pipeline([("scaler", MinMaxScaler((-1, 1)))]),
        "RecoPho_r9": Pipeline(
            [
                (
                    "log_trans",
                    FunctionTransformer(
                        lambda x: np.log(x + 1e-2), inverse_func=lambda x: np.exp(x) - 1e-2
                    ),
                ),
                (
                    "arctan_trans",
                    FunctionTransformer(
                        lambda x: np.arctan(x * 10 - 0.15),
                        inverse_func=lambda x: (np.tan(x) + 0.15) / 10,
                    ),
                ),
                ("scaler", MinMaxScaler((-1, 1))),
            ]
        ),
        "RecoPho_sieie": Pipeline(
            [
                (
                    "log_trans",
                    FunctionTransformer(
                        lambda x: np.log(x * 10 + 1e-1),
                        inverse_func=lambda x: (np.exp(x) - 1e-1) / 10,
                    ),
                ),
                (
                    "arctan_trans",
                    FunctionTransformer(
                        lambda x: np.arctan(x - 1.25),
                        inverse_func=lambda x: (np.tan(x) + 1.25),
                    ),
                ),
                ("scaler", MinMaxScaler((-1, 1))),
            ]
        ),
        "RecoPho_energyErr": Pipeline(
            [
                (
                    "log_trans",
                    FunctionTransformer(
                        lambda x: np.log1p(x), inverse_func=lambda x: np.expm1(x)
                    ),
                ),
                ("scaler", MinMaxScaler((0, 1))),
            ]
        ),
        "RecoPhoGenPho_ptratio": Pipeline(
            [
                (
                    "arctan_trans",
                    FunctionTransformer(
                        lambda x: np.arctan(x * 10 - 10),
                        inverse_func=lambda x: (np.tan(x) + 10) / 10,
                    ),
                ),
                ("scaler", MinMaxScaler((-1, 1))),
            ]
        ),
        "RecoPhoGenPho_deltaeta": Pipeline(
            [
                (
                    "arctan_trans",
                    FunctionTransformer(
                        lambda x: np.arctan(x * 100),
                        inverse_func=lambda x: (np.tan(x)) / 100,
                    ),
                ),
                ("scaler", MinMaxScaler((-1, 1))),
            ]
        ),
        "RecoPho_s4": Pipeline(
            [
                (
                    "log_trans",
                    FunctionTransformer(
                        lambda x: np.log1p(x), inverse_func=lambda x: np.expm1(x)
                    ),
                ),
                ("scaler", MinMaxScaler((0, 1))),
            ]
        ),
        "RecoPho_sieip": Pipeline([("scaler", MinMaxScaler((-1, 1)))]),
        "RecoPho_x_calo": Pipeline([("scaler", MinMaxScaler((-1, 1)))]),
        "RecoPho_hoe": Pipeline(
            [
                ('scaler', MinMaxScaler((0.1, 5))),
                ('log', FunctionTransformer(lambda x: np.log(x + 1e-3), inverse_func=lambda x: np.exp(x) - 1e-3)),
                ('scaler2', MinMaxScaler((0, 10))),
                ('displacer', Displacer(mask_lower_bound=0.1, where_to_displace=1)),
                ('smearer', Smearer('uniform', mask_upper_bound=1.0)),
                ('displacer2', Displacer(mask_lower_bound=1, where_to_displace=0.5)),
                ('scaler3', MinMaxScaler((-1., 1.))),
            ]
        ),
        "RecoPho_mvaID": Pipeline([("none", None)]),
        "RecoPho_eCorr": Pipeline([("scaler", MinMaxScaler((-1, 1)))]),
        "RecoPho_pfRelIso03_all": Pipeline(
            [
                (
                    "add",
                    FunctionTransformer(
                        lambda x: x + 1e-3, inverse_func=lambda x: x - 1e-3
                    ),
                ),
                ("box-cox", PowerTransformer(method="box-cox", standardize=False)),
                ("scaler", MinMaxScaler((0, 1))),
                ("displacer", Displacer(mask_lower_bound=0.01, where_to_displace=1)),
                ("smearer", Smearer("uniform", mask_upper_bound=1.0)),
                ("displacer_2", Displacer(mask_lower_bound=1.0, where_to_displace=0.5)),
                ("scaler_2", MinMaxScaler((-1, 1))),
            ]
        ),
        "RecoPho_pfRelIso03_chg": Pipeline(
            [
                (
                    "add",
                    FunctionTransformer(
                        lambda x: x + 1e-3, inverse_func=lambda x: x - 1e-3
                    ),
                ),
                ("box-cox", PowerTransformer(method="box-cox", standardize=False)),
                ("scaler", MinMaxScaler((0, 1))),
                ("displacer", Displacer(mask_lower_bound=0.01, where_to_displace=1)),
                ("smearer", Smearer("uniform", mask_upper_bound=1.0)),
                ("displacer_2", Displacer(mask_lower_bound=1.0, where_to_displace=0.5)),
                ("scaler_2", MinMaxScaler((-1, 1))),
            ]
        ),
        "RecoPho_esEffSigmaRR": Pipeline(
            [
                (
                    "qt",
                    QuantileTransformer(
                        n_quantiles=1000, output_distribution="normal", random_state=0
                    ),
                ),
                ("scaler", MinMaxScaler((0, 1))),
                ("displacer", Displacer(mask_lower_bound=0.5, where_to_displace=1)),
                ("smearer", Smearer("uniform", mask_upper_bound=0.5)),
                ("displacer_2", Displacer(mask_lower_bound=1, where_to_displace=0.5)),
            ]
        ),
    }
}


def main():
    files = "preprocessing/extracted_photons/*.parquet"
    output = "preprocessing/preprocessed_photons"
    fig_output = "preprocessing/preprocessed_photons/figures"
    cluster = distributed.LocalCluster(n_workers=32, threads_per_worker=1)
    client = distributed.Client(cluster)
    ddf = dd.read_parquet(files, engine="fastparquet")
    limit = 5000000
    cond_variables = [
        v for v in ddf.columns if any([v.startswith(w) for w in ["Gen", "PU"]])
    ]
    cond_variables += ["ClosestGenJet_mass", "ClosestGenJet_pt"]
    target_variables = [v for v in pipelines["pipe0"] if v.startswith("Reco")]
    all_variables = cond_variables + target_variables

    # make empty dataframe
    df = pd.DataFrame(columns=all_variables)

    # fill dataframe
    print("Filling dataframe")
    for part in ddf.to_delayed():
        print(len(df))
        if len(df) > limit:
            break
        df = pd.concat([df, part.compute()[all_variables]])

    # preprocess
    print("Preprocessing")
    for pipe in pipelines.keys():
        print(pipe)
        for v in pipelines[pipe].keys():
            # plot untransformed and transformed side by side
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].hist(df[v], bins=100, range=original_ranges[v])
            trans_arr = pipelines[pipe][v].fit_transform(df[v].values.reshape(-1, 1))
            ax[1].hist(trans_arr, bins=100)
            ax[0].set_xlabel(v)
            ax[1].set_xlabel(v)
            ax[0].set_title("Original")
            ax[1].set_title("Transformed")
            for format in ["png", "pdf"]:
                fig.savefig(os.path.join(fig_output, pipe + "_" + v + "." + format))

    # split into train, test, val in 80, 20
    train, test = train_test_split(df, test_size=0.2)
    print(len(train), len(test))

    # dump to parquet
    for name, df in zip(["train", "test"], [train, test]):
        df.to_parquet(os.path.join(output, name + ".parquet"), engine="fastparquet")

    # dump pipelines
    with open(os.path.join(output, "pipelines.pkl"), "wb") as f:
        cloudpickle.dump(pipelines, f)


if __name__ == "__main__":
    main()
