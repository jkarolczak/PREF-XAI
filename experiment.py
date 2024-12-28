import torch

from lib.config import churn
from lib.data import load
from lib.model import MLPWithEmbedding, train
from lib.optimization import solve
from lib.rules import generate_rules
from lib.tools import set_seed

DATASET_NAME = "churn"
VERBOSE = True
RETRAIN = False


def main() -> None:
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = load(DATASET_NAME)

    model = MLPWithEmbedding(
        input_dim=dataset.train.x.shape[1],
        cat_idxs=dataset.cat_idxs,
        cat_dims=dataset.cat_dims,
        **churn.model
    ).to(device=device)

    if RETRAIN:
        train(model=model, dataset=dataset, device=device, verbose=VERBOSE, **churn.train)
        torch.save(model.state_dict(), f"models/{DATASET_NAME}.pth")
    else:
        model.load_state_dict(torch.load(f"models/{DATASET_NAME}.pth", weights_only=True))

    y_hat = (model(dataset.train.as_torch(device=device)[0])[:, 1] > churn.train.threshold).cpu().int().numpy()
    rules = generate_rules(dataset, y_hat, **churn.rules)

    print(solve(rules, [19, 6, 1, 2, 12, 13, 14, 15, 16, 17, 18, 100, 24, 421, 523, 342, 5295]))


if __name__ == "__main__":
    main()
