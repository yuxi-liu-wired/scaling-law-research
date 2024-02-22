from math import log2, sqrt
import os
from scipy.special import erfc
import matplotlib.pyplot as plt


## See the NIST SP800-22r1a document for the definition of the Maurer's Universal Statistical Test


def p_value(average, L, K):
    if L == 6:
        mean, var = 5.2177052, 2.954
    if L == 7:
        mean, var = 6.1962507, 3.125
    c = 0.7 - 0.8 / L + (4 + 32 / L) * K ** (-3 / L) / 15
    sigma = c * sqrt(var / K)
    return erfc(abs(average - mean) / (sqrt(2) * sigma))


def maurer_test(Q, L, x):
    blocks = [x[i : i + L] for i in range(0, len(x) - len(x) % L, L)]
    last_occurrence = {format(i, "0" + str(L) + "b"): 0 for i in range(2**L)}
    K = len(x) // L - Q
    sum_total = 0
    num_terms = 0

    for i, block in enumerate(blocks, 1):
        if i > Q:
            old_i = last_occurrence[block]
            if old_i:
                sum_total += log2(i - old_i)
                num_terms += 1
        last_occurrence[block] = i

    average = sum_total / num_terms
    return p_value(average, L, K)


model_id = "gpt2-xl"
with open(f"wiki_arithmetic_code_{model_id}_test_1.txt", "r", encoding="utf8") as f:
    binary_sequence = f.read()
my_p_values = []
segments = 5
for i in range(segments):
    seglength = len(binary_sequence) // segments
    my_p_values.append(
        maurer_test(640, 6, binary_sequence[i * seglength : (i + 1) * seglength])
    )
print(my_p_values)
plt.hist(my_p_values)
plt.show()

# examples
# from tqdm import trange
# L = 7
# Q = 1280

# averages = []
# for i in trange(500):
#   binary_sequence = ''.join(format(byte, '08b') for byte in os.urandom(125000))
#   averages.append(maurer_test(Q, L, binary_sequence))
# plt.hist(averages)
