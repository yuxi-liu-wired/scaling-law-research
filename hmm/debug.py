import wandb
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
from random import randint
import numpy as np

ln2 = np.log(2)

# --------------------------------------------------------------------------------
# Load the dataset
# --------------------------------------------------------------------------------

# allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,")
allowed_chars = set("abcdefghijklmnopqrstuvwxyz .,")
le = LabelEncoder()
le.fit(list(allowed_chars))

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

class ConvergenceCheck:
    def __init__(self, bpc_threshold, n_epochs_min):
        self.bpc_threshold = bpc_threshold
        self.previous_bpcs = []
        self.delta_bpcs = []
        self.n_epochs_min = n_epochs_min
        self.converged = False

    def update(self, train_score):
        self.previous_bpcs.append(train_score)
        if len(self.previous_bpcs) > 1:
            self.delta_bpcs.append(self.previous_bpcs[-2] - self.previous_bpcs[-1])
            delta_bpc = self.delta_bpcs[-1]
            median_delta_bpc = np.median(self.delta_bpcs[:self.n_epochs_min])
            self.converged = delta_bpc < median_delta_bpc/2 \
                         and delta_bpc < self.bpc_threshold \
                         and len(self.previous_bpcs) > self.n_epochs_min
            print(f"Converged: {self.converged}"
                f"Delta BPC: {delta_bpc}"
                f"Median Delta BPC: {median_delta_bpc}"
                f"Previous Delta BPCs: {self.delta_bpcs}")
        
        else:
            self.converged = False
    

def train(config=None):
    n_outputs = len(allowed_chars)

    hmm_configuration = {
        "n_iter": 1,
        "params": "ste",
        "init_params": "ste",
        "verbose": False,
    }

    train_size = 160_000
    n_components = 3
    bpc_threshold = 0.005
    n_epochs_max = 1000
    n_epochs_min = 10
    idx = 807
    convergence_check = ConvergenceCheck(bpc_threshold, n_epochs_min)
    
    dataset_name = "wikitext-103-truncated"
    dataset = {}
    for item in ["train", "valid"]:
        with open(
            f"{dataset_name}/wiki_shannonfied.{item}.txt", "r", encoding="utf-8"
        ) as f:
            long_text = f.read()
            dataset[item] = [[char] for char in long_text]

    # Transform the characters into numerical labels
    X_train = le.transform(dataset["train"][:train_size]).reshape(-1, 1)
    X_val = le.transform(dataset["valid"]).reshape(-1, 1)
    # X_test = le.transform(dataset["test"]).reshape(-1, 1)

    hmm_configuration.update(
        {
            "tol": train_size * (bpc_threshold * ln2),
            "n_components": n_components,
        }
    )
    model = hmm.CategoricalHMM(random_state=idx, **hmm_configuration)

    for epoch in range(n_epochs_max):
        model.fit(X_train)
        model.init_params = ""

        # Compute scores
        train_score = -ln2 * model.score(X_train) / train_size
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
            f"Train: {train_score:.3f} bits/char, Val: {val_score:.3f} bits/char\n"
        )


        convergence_check.update(train_score)
        if convergence_check.converged:
            break

    result = {
        "sample_text": sample_text(model, 1000),
    }
    print(result["sample_text"])


if __name__ == "__main__":
    train()
