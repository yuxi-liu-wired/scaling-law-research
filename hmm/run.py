from datasets import load_dataset
import wandb
import numpy as np
from hmmlearn import hmm
import re
from sklearn.preprocessing import LabelEncoder


def sample_text(model, n_samples):
    """Sample text from the model.
    Args:
        model: The HMM model
        n_samples: The number of samples to generate
    Returns:
        The sampled text.
    """
    observations, _states = model.sample(n_samples)
    observations_flat = observations.flatten()
    decoded_characters = le.inverse_transform(observations_flat)
    return ''.join(decoded_characters)

# --------------------------------------------------------------------------------
# Load the dataset
# --------------------------------------------------------------------------------

dataset = load_dataset("wikitext", "wikitext-103-v1")
train_set = dataset['train'][:2*10**4]
val_set = dataset['validation']
test_set = dataset['test']

# allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,")
allowed_chars = set("abcdefghijklmnopqrstuvwxyz .,")

def preprocess_text(dataset):
    long_text = ' '.join(dataset['text'])
    long_text = re.sub(r"<unk>", "", long_text)
    long_text = ''.join([char.lower() for char in long_text if char.lower() in allowed_chars])
    long_text = re.sub(r"\s+", " ", long_text)
    return [[char] for char in long_text]

# Preprocess the train and test sets
train_chars = preprocess_text(train_set)
val_chars = preprocess_text(val_set)
test_chars = preprocess_text(test_set)
del train_set, val_set, test_set, dataset

le = LabelEncoder()
# Use numpy to efficiently concatenate lists
le.fit(list(allowed_chars))

# Transform the characters into numerical labels
X_train = le.transform(train_chars).reshape(-1, 1)
X_val = le.transform(val_chars).reshape(-1, 1)
X_test = le.transform(test_chars).reshape(-1, 1)
del train_chars, val_chars, test_chars

# --------------------------------------------------------------------------------
# Grid search
# --------------------------------------------------------------------------------

# Define the grid search parameters
train_sizes = [i*10**4 for i in [2, 4, 8, 16, 32, 64, 128]]
n_components_list = [3, 5, 9, 16, 27, 47, 80, 140]

n_runs = 1

hmm_configuration = {
    "n_iter": 200,
    "init_params": 'ste',
    "verbose": True
}

wandb.init(project="english_hmm", entity="yuxiliu1995", group="GridSearch")
step_counter = 0

for train_size in train_sizes:
    for n_components in n_components_list:
        for _ in range(n_runs):
            hmm_configuration.update({
                "tol": train_size * (0.005 * np.log(2)),
                "n_components": n_components
            })
            idx = np.random.randint(0, 2**15-1)
            # Initialize and fit the model
            model = hmm.CategoricalHMM(random_state=idx, **hmm_configuration)
            model.fit(X_train[:train_size])

            # Compute scores
            train_score = -np.log(2) * model.score(X_train[:train_size]) / train_size
            val_score = -np.log(2) * model.score(X_val) / len(X_val)
            aic_score = model.aic(X_val)
            bic_score = model.bic(X_val)

            # Store or print the results
            result = {
                "train_size": train_size,
                "random_state": idx,
                "n_components": n_components,
                "train_score": train_score,
                "val_score": val_score,
                "aic_score": aic_score,
                "bic_score": bic_score,
                "step": step_counter,
                "sample_text": sample_text(model, 1000)
            }
            
            print(f'Train Size: {train_size}, Random State: {idx}, Components: {n_components}\n'
                  f'\tTrain: {train_score:.3f} bits/char, Val: {val_score:.3f} bits/char\n'
                  f'\tAIC: {aic_score:.3f}, BIC: {bic_score:.3f}'
                  f'\n' + result["sample_text"] + '\n')

            wandb.log(result, commit=True)
            step_counter += 1
