import torch
from torch.utils.data import Dataset
import pandas as pd


class PhotonDataset(Dataset):
    def __init__(
        self,
        parquet_file,
        context_variables,
        target_variables,
        device=None,
        rows=None,
    ):
        self.parquet_file = parquet_file
        self.context_variables = context_variables
        self.target_variables = target_variables
        self.all_variables = context_variables + target_variables
        data = pd.read_parquet(
            parquet_file, columns=self.all_variables, engine="fastparquet"
        )
        if rows is not None:
            data = data.iloc[:rows]
        self.target = data[target_variables].values
        self.context = data[context_variables].values
        if device is not None:
            self.target = torch.tensor(self.target, dtype=torch.float32).to(device)
            self.context = torch.tensor(self.context, dtype=torch.float32).to(device)

    def __len__(self):
        assert len(self.context) == len(self.target)
        return len(self.target)

    def __getitem__(self, idx):
        return self.context[idx], self.target[idx]