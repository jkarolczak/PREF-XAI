from lib.data import TDatasetName


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


breast_cancer = DotDict({
    "model": DotDict({
        "hidden_dims": [32, 32],
        "output_dim": 2,
        "embedding_dim": 2
    }),
    "train": DotDict({
        "batch_size": 256,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 200,
        "pos_weight": [1.0, 2.0],
        "threshold": 0.5,
    }),
    "rules": DotDict({
        "n_bins": 2,
        "support_threshold": 0.0001,
        "confirm_measure": "S",
        "confirm_threshold": 0.00,
        "confidence_threshold": 1.0
    }),
    "expert": DotDict({
        "func": "diversify_by_properties",
        "n_rules": 10,
        "columns_preferences": [i for i in range(32)]
    })
})

churn = DotDict({
    "model": DotDict({
        "hidden_dims": [32, 32],
        "output_dim": 2,
        "embedding_dim": 2
    }),
    "train": DotDict({
        "batch_size": 1024,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "epochs": 150,
        "pos_weight": [1.0, 3.5],
        "threshold": 0.9,
    }),
    "rules": DotDict({
        "n_bins": 4,
        "support_threshold": 0.0001,
        "confirm_measure": "S",
        "confirm_threshold": 0.00,
        "confidence_threshold": 1.0
    }),
    "expert": DotDict({
        "func": "natural",
        "n_rules": 10,
        "columns_preferences": [9, 8, 2, 4, 3, 5, 6, 7, 1, 0]
    })
})

churn_telco = DotDict({
    "model": DotDict({
        "hidden_dims": [32, 32],
        "output_dim": 2,
        "embedding_dim": 2
    }),
    "train": DotDict({
        "batch_size": 512,
        "learning_rate": 3e-3,
        "weight_decay": 1e-5,
        "epochs": 150,
        "pos_weight": [1.0, 9.0],
        "threshold": 0.7,
    }),
    "rules": DotDict({
        "n_bins": 3,
        "support_threshold": 0.0015,
        "confirm_measure": "N",
        "confirm_threshold": 0.0,
        "confidence_threshold": 1.0
    }),
    "expert": DotDict({
        "func": "natural",
        "n_rules": 10,
        "columns_preferences": [13, 0, 2, 5, 7, 9, 11, 6, 8, 10, 12, 3, 4, 1]
    })
})


def get_config(name: TDatasetName) -> DotDict:
    return globals()[name]
