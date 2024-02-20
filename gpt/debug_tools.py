import numpy as np
import torch


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
