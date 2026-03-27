import numpy as np
import pandas as pd
from lib.optimization.prus import Prus

from lib.dms.interface_dm import IDM


class RandomDM(IDM):
    def __init__(self, rules: pd.DataFrame, n_reference_rules: int, *args, **kwargs) -> None:
        super().__init__(rules, n_reference_rules, *args, **kwargs)

        self.weights = np.random.rand(len(self.rules.columns))
        self.prus = Prus.from_np(self.weights, self.rules)

        self._reference_instances = np.random.choice(rules.index.tolist(), size=n_reference_rules, replace=False).tolist()
        self._reference_ranking = self.prus.create_ranking(rules.iloc[self._reference_instances]).index.tolist()
        self._attribute_ranking = list(np.argsort(-self.weights[:len(self.rules.columns)]).tolist())

    def create_ranking(self, rules: pd.DataFrame) -> pd.DataFrame:
        return self.prus.create_ranking(rules)
