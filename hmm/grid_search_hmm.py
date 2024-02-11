from datasets import load_dataset
import wandb
import numpy as np
from hmmlearn import hmm
import re
from sklearn.preprocessing import LabelEncoder


# --------------------------------------------------------------------------------
# Load the dataset
# --------------------------------------------------------------------------------

# allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,")
allowed_chars = set("abcdefghijklmnopqrstuvwxyz .,")
le = LabelEncoder()
le.fit(list(allowed_chars))

dataset_name = "wikitext-103-truncated"
dataset = {}
for item in ["train", "test", "valid"]:
    with open(
        f"{dataset_name}/wiki_shannonfied.{item}.txt", "r", encoding="utf-8"
    ) as f:
        long_text = f.read()
        dataset[item] = [[char] for char in long_text]

# Transform the characters into numerical labels
X_train = le.transform(dataset["train"]).reshape(-1, 1)
X_val = le.transform(dataset["valid"]).reshape(-1, 1)
X_test = le.transform(dataset["test"]).reshape(-1, 1)

# --------------------------------------------------------------------------------
# Using trained model
# --------------------------------------------------------------------------------


def sample_text(model, n_samples):
    """Sample text from the model.
    Args:
        model: The HMM model
        n_samples: The number of samples to generate
    Returns:
        The sampled text.
    """
    observations, _ = model.sample(n_samples)
    observations_flat = observations.flatten()
    decoded_characters = le.inverse_transform(observations_flat)
    return "".join(decoded_characters)


# --------------------------------------------------------------------------------
# Grid search
# --------------------------------------------------------------------------------


def parameter_count(n_components, n_outputs):
    n = n_components
    return (n - 1) + n * (n - 1) + n * (n_outputs - 1)


def run_grid_search(config):
    """Runs a grid search over the training sizes and number of components for HMM.

    Args:
        train_sizes (List[int]): The list of training sizes to use.
        n_components_list (List[int]): The list of numbers of components to use.
        n_runs (int, optional): Number of runs. Defaults to 1.
        bpc_threshold (float, optional):
            Assume that the HMM converges when each EM step improves by less than
            this many bits per character (bpc). Defaults to 0.005.
    """
    n_outputs = len(allowed_chars)

    hmm_configuration = {"n_iter": 1000, "init_params": "ste", "verbose": True}

    with wandb.init(config=config):
        config = wandb.config  # If we run the conductor-agent search.

        train_size = config.train_size
        n_components = config.n_components
        bpc_threshold = config.bpc_threshold
        hmm_configuration.update(
            {
                "tol": train_size * (bpc_threshold * np.log(2)),
                "n_components": n_components,
            }
        )
        idx = np.random.randint(0, 2**15 - 1)
        # Initialize and fit the model
        model = hmm.CategoricalHMM(random_state=idx, **hmm_configuration)
        model.fit(X_train[:train_size])

        # Compute scores
        train_score = -np.log(2) * model.score(X_train[:train_size]) / train_size
        val_score = -np.log(2) * model.score(X_val) / len(X_val)
        aic_score = model.aic(X_val)
        bic_score = model.bic(X_val)
        param_count = parameter_count(n_components=n_components, n_outputs=n_outputs)

        # Store or print the results
        result = {
            "train_size": train_size,
            "random_state": idx,
            "n_components": n_components,
            "train_score": train_score,
            "val_score": val_score,
            "aic_score": aic_score,
            "bic_score": bic_score,
            "param_count": param_count,
            "sample_text": sample_text(model, 1000),
        }

        print(
            f"Train Size: {train_size}, Components: {n_components}, Params: {param_count}\n"
            f"\tTrain: {train_score:.3f} bits/char, Val: {val_score:.3f} bits/char\n"
            f"\n" + result["sample_text"] + "\n"
        )

        wandb.log(result, commit=True)
