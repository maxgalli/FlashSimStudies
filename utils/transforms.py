import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


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
        """ Count occurrences of each unique value in arr
        Returns a dictionary with keys=unique values and values=counts
        """
        unique_values = np.unique(arr).astype("int64")
        counts = np.bincount(arr.astype("int64")[:, 0])
        dct = dict(zip(unique_values, counts[unique_values]))
        # order by key
        dct = dict(sorted(dct.items()))
        return dct

    def find_closest_numbers(self, sample, numbers):
        """ Given a sample and a list of numbers, find the closest number in the list for each element of the sample
        Returns an array of the same shape as sample
        """
        closest_indices = np.argmin(
            np.abs(sample[:, 0] - numbers[:, np.newaxis]), axis=0
        )
        return numbers[closest_indices]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()

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
        X = X.copy()
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
        X = X.copy()
        self.mask = self.apply_mask(X).reshape(X.shape)
        self.minimum = np.min(X[self.mask])
        X_transformed = np.where(
            self.mask, X - self.minimum + self.where_to_displace, X
        )
        return X_transformed

    def inverse_transform(self, X, y=None):
        X = X.copy()
        self.mask = self.apply_mask(X).reshape(X.shape)
        return np.where(
            self.mask, X + self.minimum - self.where_to_displace, X
        )

        
class IsoTransformerBC(TransformerMixin, BaseEstimator):
    """
    Transform the isolation variables with a box-cox transformation for values > 0 and a normal distribution for values <= 0.
    """
    def __init__(self, loc=0., scale=0.1):
        self.loc = loc
        self.scale = scale
    
    def fit(self, X, y=None):
        non_zero_indices = np.where(X > 0)[0]
        x_non_zero = X[non_zero_indices]
        self.bc = PowerTransformer(method="box-cox").fit(x_non_zero.reshape(-1, 1))
        x_non_zero_bc = self.bc.transform(x_non_zero.reshape(-1, 1))
        self.min_bc = np.min(x_non_zero_bc)
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        zero_indices = np.where(X <= 0)[0]
        nonzero_indices = np.where(X > 0)[0]
        # replace 0s with values sampled from gaussian
        X[zero_indices] = np.random.normal(loc=self.loc, scale=self.scale, size=len(zero_indices)).reshape(-1, 1)
        # shift the rest wiht boxcox
        X[nonzero_indices] = self.bc.transform(X[nonzero_indices])
        # now set the max of lognormal to 0 and the min of boxcox to 0
        X[zero_indices] -= np.max(X[zero_indices])
        X[nonzero_indices] -= self.min_bc
        return X
        
    def inverse_transform(self, X, y=None):
        X = X.copy()
        zero_indices = np.where(X <= 0)[0]
        nonzero_indices = np.where(X > 0)[0]
        # transform back the boxcox
        X[nonzero_indices] += self.min_bc
        X[nonzero_indices] = self.bc.inverse_transform(X[nonzero_indices])
        # replace values less than 0 with 0
        X[zero_indices] = 0.
        return X

        
class IsoTransformerLNorm(TransformerMixin, BaseEstimator):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X = X.copy()
        zero_indices = np.where(X <= 0)[0]
        nonzero_indices = np.where(X > 0)[0]
        # replace 0s with values sampled from triangular distribution
        X[zero_indices] = -np.random.lognormal(mean=0.0001, sigma=0.1, size=len(zero_indices)).reshape(-1, 1)
        # shift the rest
        X[nonzero_indices] = np.log1p(X[nonzero_indices])
        return X
        
    def inverse_transform(self, X, y=None):
        X = X.copy()
        zero_indices = np.where(X <= 0)[0]
        nonzero_indices = np.where(X > 0)[0]
        # expm1 the rest
        X[nonzero_indices] = np.expm1(X[nonzero_indices])
        # replace values less than 0 with 0
        X[zero_indices] = 0.
        return X