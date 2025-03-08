from __future__ import annotations

import warnings
from abc import ABC
from copy import deepcopy
from dataclasses import (dataclass,
                         field)
from pathlib import Path
from typing import (Literal,
                    TypeAlias)

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import (OrdinalEncoder,
                                   StandardScaler)

from lib import RANDOM_SEED

TDatasetName: TypeAlias = Literal[
    "breast_cancer",
    "churn",
    "churn_telco"
]


@dataclass
class AbstractDataclass(ABC):
    def __new__(cls, *args, **kwargs) -> object:
        if cls == AbstractDataclass or cls.__bases__[0] == AbstractDataclass:
            raise TypeError("Cannot instantiate abstract class.")
        return super().__new__(cls)


@dataclass
class Subset:
    def __init__(self, x: pd.DataFrame, y: pd.Series, parent: IDataset) -> None:
        self.x = x
        self.y = y
        self.parent = parent

    def __len__(self) -> int:
        return len(self.x)

    def length(self) -> int:
        return len(self)

    def as_encoded(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.parent.scale(self.parent.encode(self.x)), pd.get_dummies(self.y)

    def as_torch(self, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.as_encoded()
        if device is None:
            device = torch.device("cpu")
        return (torch.tensor(x.values.astype(np.float32)).to(device),
                torch.tensor(y.values.astype(np.float32)).to(device))


@dataclass
class IDataset(AbstractDataclass):
    name: str
    x: pd.DataFrame
    y: pd.Series
    cat_idxs: list[int] = field(default_factory=list)
    cat_dims: list[int] | None = None
    num_idxs: list[int] | None = None
    file_path: Path | None = None
    train: Subset | None = None
    test: Subset | None = None
    encoders: dict[int, OrdinalEncoder] = field(default_factory=dict)
    scalers: dict[int, StandardScaler] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=RANDOM_SEED,
                                                            stratify=self.y)
        self.train = Subset(x_train, y_train, self)
        self.test = Subset(x_test, y_test, self)
        self.num_idxs = [i for i in range(len(self.x.columns)) if i not in self.cat_idxs]
        self.cat_dims = [self.x.iloc[:, idx].nunique() for idx in self.cat_idxs]
        self.encoders = {i: OrdinalEncoder().fit(self.train.x.iloc[:, i].values.reshape(-1, 1)) for i in self.cat_idxs}
        self.scalers = {i: StandardScaler().fit(self.train.x.iloc[:, i].values.reshape(-1, 1)) for i in self.num_idxs}

    def _transform(self, transforms: dict[int, callable], x: pd.DataFrame | None = None) -> pd.DataFrame:
        if x is None:
            x = self.x
        x = x.copy()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i, transform in transforms.items():
                x.iloc[:, i] = transform.transform(x.iloc[:, i].values.reshape(-1, 1)).flatten()

        return x

    def encode(self, x: pd.DataFrame | None = None) -> pd.DataFrame:
        return self._transform(self.encoders, x)

    def scale(self, x: pd.DataFrame | None = None) -> pd.DataFrame:
        return self._transform(self.scalers, x)

    def __repr__(self) -> str:
        return f"{self.name.capitalize()}({len(self.x)} rows\nx:\n{self.x.head(3)}\ny:\n{self.y.head(3)})"

    def to_cv(self, n_folds: int = 10) -> list[IDataset]:
        all_idcs = np.arange(len(self.x))
        np.random.seed(RANDOM_SEED)
        np.random.shuffle(all_idcs)
        folds_idcs = np.array_split(all_idcs, n_folds)
        splits = []
        for idcs in folds_idcs:
            churn = deepcopy(self)
            churn.train = Subset(self.x.iloc[idcs], self.y.iloc[idcs], churn)
            churn.test = Subset(self.x.iloc[~idcs], self.y.iloc[~idcs], churn)
            splits.append(churn)
        return splits


class BreastCancer(IDataset):
    def __init__(self) -> None:
        df = pd.read_csv("data/breast_cancer.csv", index_col=False, low_memory=False)
        df = df.drop(columns=["id"])
        x = df.drop(columns="diagnosis")
        y = (df["diagnosis"] == "M").astype(int)
        super().__init__(name="breast_cancer", x=x, y=y)


class Churn(IDataset):
    def __init__(self) -> None:
        df = pd.read_csv("data/churn.csv", index_col=0, low_memory=False)
        df = df.drop(columns=["CustomerId", "Surname"])
        x = df.drop(columns="Exited")
        y = df["Exited"]
        super().__init__(name="churn", x=x, y=y, cat_idxs=[1, 2, 6, 7, 8])


class ChurnTelco(IDataset):
    def __init__(self) -> None:
        df = pd.read_csv("data/churn_telco.csv", index_col=False, low_memory=False, sep=";", decimal=",")
        df = df.drop(
            columns=["state", "phone number", "total day calls", "total eve calls", "total night calls", "total intl calls"])
        x = df.drop(columns="churn")
        y = df["churn"]
        super().__init__(name="churn_telco", x=x, y=y, cat_idxs=[1, 2, 3])


def load(name: TDatasetName) -> IDataset:
    cls_name = "".join([w.capitalize() for w in name.split("_")])
    return globals()[cls_name]()
