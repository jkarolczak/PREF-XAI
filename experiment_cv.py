import click
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from tqdm.cli import tqdm

from lib.config import get_config
from lib.data import (load,
                      TDatasetName)
from lib.model import (MLPWithEmbedding,
                       train)
from lib.tools import set_seed


@click.command()
@click.argument("dataset_name", type=str, default="churn")
def main(dataset_name: TDatasetName) -> None:
    config = get_config(dataset_name)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load(dataset_name)

    folds = dataset.to_cv(n_folds=10)

    balanced_accuracy = []
    accuracy = []

    for ds in tqdm(folds, desc="Folds"):
        model = MLPWithEmbedding(
            input_dim=dataset.train.x.shape[1],
            cat_idxs=dataset.cat_idxs,
            cat_dims=dataset.cat_dims,
            **config.model
        ).to(device=device)

        train(model=model, dataset=dataset, device=device, verbose=False, **config.train)

        proba_valid = model(ds.test.as_torch(device=device)[0])[:, 1].detach().cpu().numpy()
        y_hat_valid = (proba_valid > config.train.threshold).astype(int)
        balanced_accuracy.append(balanced_accuracy_score(ds.test.y, y_hat_valid).item())
        accuracy.append(accuracy_score(ds.test.y, y_hat_valid))

    print(f"Balanced accuracy: {balanced_accuracy}")
    print(f"{np.mean(balanced_accuracy).item():.4f} ± {np.std(balanced_accuracy).item():.4f}")

    print(f"Accuracy: {accuracy}")
    print(f"{np.mean(accuracy).item():.4f} ± {np.std(accuracy).item():.4f}")


if __name__ == "__main__":
    main()
