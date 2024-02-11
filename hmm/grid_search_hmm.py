import wandb
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from random import randint

ln2 = 0.69314718

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


def train(config=None):
    """Runs a grid search over the training sizes and number of components for HMM.

    Args:
        train_sizes (List[int]): The list of training sizes to use.
        n_components_list (List[int]): The list of numbers of components to use.
        n_runs (int, optional): Number of runs. Defaults to 1.
        bpc_threshold (float, optional):
            Assume that the HMM converges when each EM step improves by less than
            this many bits per character (bpc). Defaults to 0.005.
    """
    previous_bpc = 1e9
    n_outputs = len(allowed_chars)

    hmm_configuration = {
        "n_iter": 1,
        "params": "ste",
        "init_params": "ste",
        "verbose": True,
    }

    with wandb.init(config=config):
        config = wandb.config  # If we run the conductor-agent search.

        train_size = config.train_size
        n_components = config.n_components
        bpc_threshold = config.bpc_threshold
        n_epochs_max = config.n_epochs_max
        hmm_configuration.update(
            {
                "tol": train_size * (bpc_threshold * ln2),
                "n_components": n_components,
            }
        )
        idx = randint(0, 2**15)
        model = hmm.CategoricalHMM(random_state=idx, **hmm_configuration)

        for epoch in range(n_epochs_max):
            model.fit(X_train[:train_size])
            model.init_params = ""

            # Compute scores
            train_score = -ln2 * model.score(X_train[:train_size]) / train_size
            val_score = -ln2 * model.score(X_val) / len(X_val)
            param_count = parameter_count(
                n_components=n_components, n_outputs=n_outputs
            )

            # Store or print the results
            result = {
                "train_size": train_size,
                "random_state": idx,
                "n_components": n_components,
                "train_score": train_score,
                "val_score": val_score,
                "param_count": param_count,
            }

            print(
                f"Train Size: {train_size}, Components: {n_components}, Params: {param_count}\n"
                f"\tTrain: {train_score:.3f} bits/char, Val: {val_score:.3f} bits/char\n"
                f"\n" + result["sample_text"] + "\n"
            )

            wandb.log(result, commit=True)

            delta_bpc = previous_bpc - val_score
            if delta_bpc < bpc_threshold:
                break
            previous_bpc = val_score

        result = {
            "sample_text": sample_text(model, 1000),
        }
        wandb.log(result, commit=True)


if __name__ == "__main__":
    train()
