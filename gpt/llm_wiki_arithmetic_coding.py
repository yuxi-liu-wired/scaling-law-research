import numpy as np
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import argparse

# Setup argument parser
parser = argparse.ArgumentParser(description='Run model with specified ID.')
parser.add_argument('--model_id', type=str, required=True, 
                    help='Model ID to use', 
                    choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'])

parser.add_argument('--dataset_split', type=str, required=True, 
                    help='Dataset split to use', 
                    choices=['train', 'test', 'validation'])
parser.add_argument('--stride', type=int, default=512, 
                    help='Stride to use for traversing the dataset.')

args = parser.parse_args()

# Use the model_id from the command-line arguments
model_id = args.model_id
dataset_split = args.dataset_split
stride = args.stride

# --------------------------------------------------------------------------------
# Load the model
# --------------------------------------------------------------------------------

device = "cuda"
model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

# --------------------------------------------------------------------------------
# Load the dataset
# --------------------------------------------------------------------------------

allowed_chars = set("abcdefghijklmnopqrstuvwxyz .,")

dataset_name = "wikitext-103-truncated"
dataset = {}
for item in ["valid", "test"]:
    with open(
        f"{dataset_name}/wiki_shannonfied.{dataset_split}.txt", "r", encoding="utf-8"
    ) as f:
        dataset[dataset_split] = f.read()

encodings = tokenizer(dataset[dataset_split], return_tensors="pt")
encodings.input_ids = encodings.input_ids.to(device)

seq_len = encodings.input_ids.size(1)
print(f"Sequence length = {seq_len}\n")
print(tokenizer.decode(encodings.input_ids[0][seq_len-400:]))

# --------------------------------------------------------------------------------
# Calculate intervals
# --------------------------------------------------------------------------------

def logit_to_cdf(logit):
    prob = np.exp(logit - logit.max())
    cdf = np.cumsum(prob)
    cdf /= cdf.max()
    cdf = np.concatenate((np.zeros(1), cdf))
    return cdf

def logit_array_to_cdf(logit, axis, epsilon=1e-9):
    if isinstance(logit, np.ndarray):
        logit = torch.tensor(logit)

    max_logit = torch.max(logit, axis=axis, keepdims=True)[0]
    prob = torch.exp(logit - max_logit)
    prob /= prob.sum(axis=axis, keepdims=True)
    prob += epsilon
    cdf = torch.cumsum(prob, axis=axis)
    cdf /= torch.max(cdf, axis=axis, keepdims=True)[0]
    # append 0 to the beginning of the cdf along axis=axis
    shape = list(cdf.shape)
    shape[axis] = 1
    cdf = torch.concatenate((torch.zeros(shape).to(cdf.device), cdf), axis=axis)
    return cdf

def get_intervals(logits, symbols, epsilon=1e-9):
    original_shape = logits.shape
    logits = logits.reshape(-1, logits.shape[-1])
    symbols = symbols.reshape(-1)

    cdf = logit_array_to_cdf(logits, axis=1, epsilon=epsilon)

    intervals = []
    for i, symbol in enumerate(symbols):
        lower_bound = cdf[i, symbol]
        upper_bound = cdf[i, symbol + 1]

        intervals.append([lower_bound.item(), upper_bound.item()])

    # Convert intervals list to a tensor
    intervals_tensor = torch.tensor(intervals)
    intervals_tensor = intervals_tensor.reshape(original_shape[:-1] + (2,))
    return intervals_tensor

max_length = model.config.n_positions
seq_len = encodings.input_ids.size(1)

intervals_list = []

for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    if end_loc + 1 >= seq_len:
        break # Let's just throw away the tail. It's easier than dealing with the annoying tail.
    input_ids = encodings.input_ids[:, begin_loc:end_loc]
    with torch.no_grad():
        outputs = model(input_ids)

    logits = outputs.logits[0, -stride:]
    symbols = encodings.input_ids[:, end_loc+1-stride:end_loc+1]
    intervals_list.append(get_intervals(logits, symbols, epsilon=1e-7))


# --------------------------------------------------------------------------------
# Calculate arithmetic code
# --------------------------------------------------------------------------------

from arithmetic_coding import ArithmeticCode

ae = ArithmeticCode(32)
intervals = torch.cat(intervals_list, axis=0)
wiki_arithmetic_code = ae.encode_intervals(intervals)

print(f"Originally {intervals.shape[0]} characters.")
print(f"File compressed to {len(wiki_arithmetic_code)} bits, which is {len(wiki_arithmetic_code)/2**23:.2f} MB.")
print(f"The arithmetic interval has length {-torch.log2(intervals[:, 1] - intervals[:, 0]).sum():.2f} bits.")
print(f"Bit rate = {len(wiki_arithmetic_code)/intervals.shape[0]:.2f} bit/token")
print(f"         = {len(wiki_arithmetic_code)/len(dataset[dataset_split]):.2f} bit/token")

save_file = f"wiki_arithmetic_code_{model_id}_{dataset_split}_{stride}"
with open(f'{save_file}.txt', 'w', encoding="utf8") as f:
    f.write(wiki_arithmetic_code.bin)

with open(f'{save_file}.bin','wb') as f:
    wiki_arithmetic_code.tofile(f)