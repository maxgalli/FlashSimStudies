import os
import cloudpickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.base import TransformerMixin, BaseEstimator
import dask.dataframe as dd
import dask
import distributed
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class MaskMixin:
    """Mixin class for masking values in a numpy array"""

    def __init__(self, mask_lower_bound=None, mask_upper_bound=None):
        self.mask_lower_bound = mask_lower_bound
        self.mask_upper_bound = mask_upper_bound

    def apply_mask(self, arr):
        mask = np.ones(arr.shape[0], dtype=bool)
        if self.mask_lower_bound is not None:
            mask = mask & (arr >= np.asarray(self.mask_lower_bound)).reshape(-1)
        if self.mask_upper_bound is not None:
            mask = mask & (arr <= np.asarray(self.mask_upper_bound)).reshape(-1)
        return mask


class Smearer(TransformerMixin, BaseEstimator, MaskMixin):
    def __init__(self, kind, mask_lower_bound=None, mask_upper_bound=None):
        if kind not in ["gaus", "uniform"]:
            raise ValueError
        self.kind = kind
        self.mask_lower_bound = mask_lower_bound
        self.mask_upper_bound = mask_upper_bound

    def get_half_distances(self, arr):
        diffs = np.diff(arr)
        half_diffs = diffs / 2
        result = np.concatenate(
            (
                [arr[0] - half_diffs[0]],
                arr[:-1] + half_diffs,
                [arr[-1] + half_diffs[-1]],
            )
        )
        return result

    def count_occurrences(self, arr):
        unique_values = np.unique(arr).astype("int64")
        counts = np.bincount(arr.astype("int64")[:, 0])
        dct = dict(zip(unique_values, counts[unique_values]))
        # order by key
        dct = dict(sorted(dct.items()))
        return dct

    def find_closest_numbers(self, sample, numbers):
        closest_indices = np.argmin(
            np.abs(sample[:, 0] - numbers[:, np.newaxis]), axis=0
        )
        return numbers[closest_indices]

    def fit(self, X, y=None):
        self.occurrences = self.count_occurrences(X)
        self.values = np.array(list(self.occurrences.keys()))
        self.half_distances = self.get_half_distances(
            self.values
        )  # one more item wrt occurrances, values and half_widths
        self.half_widths = np.array(
            [
                np.abs(i - j)
                for i, j in zip(self.half_distances[:-1], self.half_distances[1:])
            ]
        )

        return self

    def transform(self, X, y=None):
        self.mask = self.apply_mask(X).reshape(X.shape)
        
        new_sub_arrs = []

        for idx, (number, occ) in enumerate(self.occurrences.items()):
            if self.kind == "uniform":
                smear = np.random.uniform(
                    low=self.half_distances[idx],
                    high=self.half_distances[idx + 1],
                    size=occ,
                )
            elif self.kind == "gaus":
                scale = self.half_widths[idx] / 8
                smear = np.random.normal(loc=number, scale=scale, size=occ)

            new_sub_arrs.append(smear)

        new_sub_arrs = np.concatenate(new_sub_arrs).reshape(X.shape)
        new_sub_arrs = np.sort(new_sub_arrs, axis=0)
        order = np.argsort(np.argsort(X, axis=0), axis=0).reshape(-1)
        new_sub_arrs = new_sub_arrs[order].reshape(X.shape)

        # applying smear for masked values and retaining original for others
        X_transformed = np.where(self.mask, new_sub_arrs, X)

        return X_transformed

    def inverse_transform(self, X, y=None):
        self.mask = self.apply_mask(X).reshape(X.shape)
        return np.where(
            self.mask,
            self.find_closest_numbers(X, self.values).reshape(-1, 1),
            X,
        )


class Displacer(TransformerMixin, BaseEstimator, MaskMixin):
    """Move the minimum to where_to_displace"""

    def __init__(
        self, mask_lower_bound=None, mask_upper_bound=None, where_to_displace=None
    ):
        self.mask_lower_bound = mask_lower_bound
        self.mask_upper_bound = mask_upper_bound
        self.where_to_displace = where_to_displace

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        self.mask = self.apply_mask(X).reshape(X.shape)
        self.minimum = np.min(X[self.mask])
        X_transformed = np.where(
            self.mask, X - self.minimum + self.where_to_displace, X
        )
        return X_transformed

    def inverse_transform(self, X, y=None):
        self.mask = self.apply_mask(X).reshape(X.shape)
        return np.where(
            self.mask, X + self.minimum - self.where_to_displace, X
        )


pipelines = {
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

original_ranges = {
    "GenPho_pt": (0, 300),
    "GenPho_eta": (-5, 5),
    "GenPho_phi": (-5, 5),
    "GenPho_status": (0, 45),
    "GenPhoGenEle_deltar": (0, 15),
    "ClosestGenJet_pt": (0, 300),
    "ClosestGenJet_mass": (0, 40),
    "PU_gpudensity": (0, 1),
    "PU_nPU": (0, 100),
    "PU_nTrueInt": (0, 100),
    "PU_pudensity": (0, 10),
    "PU_sumEOOT": (0, 1000),
    "PU_sumLOOT": (0, 200),
    "RecoPho_r9": (0, 2),
    "RecoPho_sieie": (0, 0.1),
    "RecoPho_energyErr": (0, 100),
    "RecoPhoGenPho_ptratio": (0, 10),
    "RecoPhoGenPho_deltaeta": (0, 3),
    "RecoPho_s4": (0, 1),
    "RecoPho_sieip": (-0.001, 0.001),
    "RecoPho_x_calo": (-150, 150),
    "RecoPho_hoe": (0, 2),
    "RecoPho_mvaID": (-1, 1),
    "RecoPho_eCorr": (0.9, 1.1),
    "RecoPho_pfRelIso03_all": (0, 3),
    "RecoPho_pfRelIso03_chg": (0, 1.5),
    "RecoPho_esEffSigmaRR": (0, 20),
}


def main():
    files = "extracted_photons/*.parquet"
    output = "preprocessed_photons"
    fig_output = "preprocessed_photons/figures"
    cluster = distributed.LocalCluster(n_workers=32, threads_per_worker=1)
    client = distributed.Client(cluster)
    ddf = dd.read_parquet(files, engine="fastparquet")
    limit = 5000000
    cond_variables = [
        v for v in ddf.columns if any([v.startswith(w) for w in ["Gen", "PU"]])
    ]
    cond_variables += ["ClosestGenJet_mass", "ClosestGenJet_pt"]
    target_variables = [v for v in pipelines if v.startswith("Reco")]
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
    for v in pipelines.keys():
        print(v)
        # plot untransformed and transformed side by side
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].hist(df[v], bins=100, range=original_ranges[v])
        df[v] = pipelines[v].fit_transform(df[v].values.reshape(-1, 1))
        ax[1].hist(df[v], bins=100)
        ax[0].set_xlabel(v)
        ax[1].set_xlabel(v)
        for format in ["png", "pdf"]:
            fig.savefig(os.path.join(fig_output, v + "." + format))

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
