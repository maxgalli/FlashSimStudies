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

cond_variables = []

pipelines = {
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
}

original_ranges = {
    "RecoPho_r9": (0, 2),
    "RecoPhoGenPho_ptratio": (0, 10),
}

def main():
    files = "extracted_photons/*.parquet"
    output = "preprocessed_photons"
    fig_output = "preprocessed_photons/figures"
    cluster = distributed.LocalCluster(n_workers=32, threads_per_worker=1)
    client = distributed.Client(cluster)
    ddf = dd.read_parquet(files, engine="fastparquet")
    limit = 5000000
    cond_variables = [v for v in ddf.columns if any([v.startswith(w) for w in ["Gen", "PU"]])]
    target_variables = list(pipelines.keys())
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
    for v in target_variables:
        # plot untransformed and transformed side by side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist(df[v], bins=100)
        df[v] = pipelines[v].fit_transform(df[v].values.reshape(-1, 1))
        ax[1].hist(df[v], bins=100)
        ax[0].set_xlabel(v)
        ax[1].set_xlabel(v)
        for format in ["png", "pdf"]:
            fig.savefig(os.path.join(fig_output, v + "." + format))
    
    # split into train, test, val in 60, 40
    train, test = train_test_split(df, test_size=0.4)
    print(len(train), len(test))
    
    # dump to parquet
    for name, df in zip(["train",  "test"], [train, test]):
        df.to_parquet(os.path.join(output, name + ".parquet"), engine="fastparquet")
    
    # dump pipelines
    with open(os.path.join(output, "pipelines.pkl"), "wb") as f:
        cloudpickle.dump(pipelines, f)
    

if __name__ == "__main__":
    main()
