import os
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, PowerTransformer, QuantileTransformer, StandardScaler
import dask.dataframe as dd
import dask
import distributed
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils.transforms import original_ranges, Smearer, Displacer, IsoTransformerBC, IsoTransformerLNorm
from utils.plots import dump_main_plot


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
    },
    "pipe1": {
        "GenPho_pt": Pipeline([("box-cox", PowerTransformer(method="box-cox"))]),
        "GenPho_eta": Pipeline([("standard", StandardScaler())]),
        "GenPho_phi": Pipeline([("standard", StandardScaler())]),
        "GenPho_status": Pipeline(
            [
                ("smearer", Smearer("uniform")),
                ("standard", StandardScaler()),
            ]
        ),
        "GenPhoGenEle_deltar": Pipeline([("standard", StandardScaler())]),
        "ClosestGenJet_pt": Pipeline([("box-cox", PowerTransformer(method="box-cox"))]),
        "ClosestGenJet_mass": Pipeline([("standard", StandardScaler())]),
        "PU_gpudensity": Pipeline([("none", None)]),
        "PU_nPU": Pipeline([("standard", StandardScaler())]),
        "PU_nTrueInt": Pipeline([("standard", StandardScaler())]),
        "PU_pudensity": Pipeline(
            [
                ("smearer", Smearer("uniform")),
                ("standard", StandardScaler()),
            ]
        ),
        "PU_sumEOOT": Pipeline([("standard", StandardScaler())]),
        "PU_sumLOOT": Pipeline([("standard", StandardScaler())]),
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
                ("standard", StandardScaler()),
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
                ("scaler", StandardScaler()),
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
                ("standard", StandardScaler()),
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
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPhoGenPho_deltaeta": Pipeline(
            [
                ("move-right", FunctionTransformer(lambda x: x + 1e-3, inverse_func=lambda x: x - 1e-3)),
                ("box-cox", PowerTransformer(method="box-cox")),
                ("standard", StandardScaler()),
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
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPho_sieip": Pipeline([("standard", StandardScaler())]),
        "RecoPho_x_calo": Pipeline([("standard", StandardScaler())]),
        "RecoPho_hoe": Pipeline(
            [
                ("iso_transform", IsoTransformerBC(loc=-1, scale=0.5)),
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPho_mvaID": Pipeline([("none", None)]),
        "RecoPho_eCorr": Pipeline([("standard", StandardScaler())]),
        "RecoPho_pfRelIso03_all": Pipeline(
            [
                ("iso_transform", IsoTransformerBC(loc=-1, scale=0.5)),
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPho_pfRelIso03_chg": Pipeline(
            [
                ("iso_transform", IsoTransformerBC(loc=-1, scale=0.5)),
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPho_esEffSigmaRR": Pipeline(
            [
                ("iso_transform", IsoTransformerLNorm()),
                ("standard", StandardScaler()),
            ]
        )
    },
    "pipe1_bis": {
        "GenPho_pt": Pipeline([("box-cox", PowerTransformer(method="box-cox"))]),
        "GenPho_eta": Pipeline([("standard", StandardScaler())]),
        "GenPho_phi": Pipeline([("standard", StandardScaler())]),
        "GenPho_status": Pipeline(
            [
                ("smearer", Smearer("uniform")),
                ("standard", StandardScaler()),
            ]
        ),
        "GenPhoGenEle_deltar": Pipeline([("standard", StandardScaler())]),
        "ClosestGenJet_pt": Pipeline([("box-cox", PowerTransformer(method="box-cox"))]),
        "ClosestGenJet_mass": Pipeline([("standard", StandardScaler())]),
        "PU_gpudensity": Pipeline([("none", None)]),
        "PU_nPU": Pipeline([("standard", StandardScaler())]),
        "PU_nTrueInt": Pipeline([("standard", StandardScaler())]),
        "PU_pudensity": Pipeline(
            [
                ("smearer", Smearer("uniform")),
                ("standard", StandardScaler()),
            ]
        ),
        "PU_sumEOOT": Pipeline([("standard", StandardScaler())]),
        "PU_sumLOOT": Pipeline([("standard", StandardScaler())]),
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
                ("standard", StandardScaler()),
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
                ("scaler", StandardScaler()),
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
                ("standard", StandardScaler()),
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
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPhoGenPho_deltaeta": Pipeline(
            [
                ("move-right", FunctionTransformer(lambda x: x + 1e-3, inverse_func=lambda x: x - 1e-3)),
                ("box-cox", PowerTransformer(method="box-cox")),
                ("standard", StandardScaler()),
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
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPho_sieip": Pipeline([("standard", StandardScaler())]),
        "RecoPho_x_calo": Pipeline([("standard", StandardScaler())]),
        "RecoPho_hoe": Pipeline(
            [
                ("iso_transform", IsoTransformerLNorm()),
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPho_mvaID": Pipeline([("none", None)]),
        "RecoPho_eCorr": Pipeline([("standard", StandardScaler())]),
        "RecoPho_pfRelIso03_all": Pipeline(
            [
                ("iso_transform", IsoTransformerLNorm()),
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPho_pfRelIso03_chg": Pipeline(
            [
                ("iso_transform", IsoTransformerLNorm()),
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPho_esEffSigmaRR": Pipeline(
            [
                ("iso_transform", IsoTransformerLNorm()),
                ("standard", StandardScaler()),
            ]
        )
    },
    "pipe1_nocontext": {
        "GenPho_pt": Pipeline([("none", None)]),
        "GenPho_eta": Pipeline([("none", None)]),
        "GenPho_phi": Pipeline([("none", None)]),
        "GenPho_status": Pipeline([("none", None)]),
        "GenPhoGenEle_deltar": Pipeline([("none", None)]),
        "ClosestGenJet_pt": Pipeline([("none", None)]),
        "ClosestGenJet_mass": Pipeline([("none", None)]),
        "PU_gpudensity": Pipeline([("none", None)]),
        "PU_nPU": Pipeline([("none", None)]),
        "PU_nTrueInt": Pipeline([("none", None)]),
        "PU_pudensity": Pipeline([("none", None)]),
        "PU_sumEOOT": Pipeline([("none", None)]),
        "PU_sumLOOT": Pipeline([("none", None)]),
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
                ("standard", StandardScaler()),
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
                ("scaler", StandardScaler()),
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
                ("standard", StandardScaler()),
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
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPhoGenPho_deltaeta": Pipeline(
            [
                ("move-right", FunctionTransformer(lambda x: x + 1e-3, inverse_func=lambda x: x - 1e-3)),
                ("box-cox", PowerTransformer(method="box-cox")),
                ("standard", StandardScaler()),
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
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPho_sieip": Pipeline([("standard", StandardScaler())]),
        "RecoPho_x_calo": Pipeline([("standard", StandardScaler())]),
        "RecoPho_hoe": Pipeline(
            [
                ("iso_transform", IsoTransformerLNorm()),
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPho_mvaID": Pipeline([("none", None)]),
        "RecoPho_eCorr": Pipeline([("standard", StandardScaler())]),
        "RecoPho_pfRelIso03_all": Pipeline(
            [
                ("iso_transform", IsoTransformerLNorm()),
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPho_pfRelIso03_chg": Pipeline(
            [
                ("iso_transform", IsoTransformerLNorm()),
                ("standard", StandardScaler()),
            ]
        ),
        "RecoPho_esEffSigmaRR": Pipeline(
            [
                ("iso_transform", IsoTransformerLNorm()),
                ("standard", StandardScaler()),
            ]
        )
    },

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

    # split into train, test, val in 80, 20
    train, test = train_test_split(df, test_size=0.2)
    print(len(train), len(test))

    # preprocess
    print("Preprocessing")
    for pipe_name in pipelines:
        print(pipe_name)
        dct = pipelines[pipe_name]
        for v, pipe in dct.items():
            print(v)
            # plot untransformed and transformed side by side
            arr = train[v].values
            arr_trans = pipe.fit_transform(arr.reshape(-1, 1))
            arr_trans_back = pipe.inverse_transform(arr_trans.reshape(-1, 1))
            fig, (ax1, ax2) = dump_main_plot(
                arr,
                arr_trans_back,
                v,
                100,
                original_ranges[v],
                labels=["Original", "Transformed Back"],    
            )
            for format in ["png"]:
                fig.savefig(os.path.join(fig_output, pipe_name + "_" + v + "." + format))
            
            # plot arr and arr_trans side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
            ax1.hist(arr, bins=100, histtype="step", range=original_ranges[v], density=True)
            ax1.set_title("Original")
            ax2.hist(arr_trans, bins=100, histtype="step", range=(np.min(arr_trans), np.max(arr_trans)), density=True)
            ax2.set_title("Transformed")
            ax1.set_xlabel(v)
            ax2.set_xlabel(v)
            ax1.set_ylabel("Normalized yield")
            for format in ["png"]:
                fig.savefig(os.path.join(fig_output, pipe_name + "_" + v + "_both." + format))

    # plot also for test
    print("plotting also for test dataset")
    for pipe_name in pipelines:
        dct = pipelines[pipe_name]
        for v, pipe in dct.items():
            arr = test[v].values
            arr_trans = pipe.transform(arr.reshape(-1, 1))
            arr_trans_back = pipe.inverse_transform(arr_trans.reshape(-1, 1))
            fig, (ax1, ax2) = dump_main_plot(
                arr,
                arr_trans_back,
                v,
                100,
                original_ranges[v],
                labels=["Original", "Transformed Back"],
            )
            for format in ["png"]:
                fig.savefig(os.path.join(fig_output, "test_" + pipe_name + "_" + v + "." + format))

    # dump to parquet
    for name, df in zip(["train", "test"], [train, test]):
        df.to_parquet(os.path.join(output, name + ".parquet"), engine="fastparquet")

    # dump pipelines
    with open(os.path.join(output, "pipelines.pkl"), "wb") as f:
        cloudpickle.dump(pipelines, f)


if __name__ == "__main__":
    main()
