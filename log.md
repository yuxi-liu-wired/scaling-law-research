# Research log

## 02-20

Got the arithmetic compression working, on about 1.2 million characters (`wiki103.test`).

| Model       | Compressed Size (bits) | Bit Rate (bit/token) | Bit Rate (bit/char) |
| ----------- | ---------------------- | ------------------- | ------------------- |
| gpt2        | 1,334,322              | 5.18                 | 1.10                |
| gpt2-medium | 1,187,137              | 4.61                 | 0.98                |
| gpt2-large  | 1,133,520              | 4.40                 | 0.93                |
| gpt2-xl     | 1,092,744              | 4.24                 | 0.90                |


Test results of the NIST test suite on `wiki_arithmetic_code_gpt2-xl_test_1`:

- PASSED - score: 0.292 - Monobit - elapsed time: 0 ms
- PASSED - score: 0.063 - Frequency Within Block - elapsed time: 1 ms
- PASSED - score: 0.959 - Runs - elapsed time: 210 ms
- PASSED - score: 0.281 - Longest Run Ones In A Block - elapsed time: 102 ms
- PASSED - score: 0.209 - Binary Matrix Rank - elapsed time: 1834 ms
- FAILED - score: 0.0 - Discrete Fourier Transform - elapsed time: 411 ms
- PASSED - score: 0.355 - Non Overlapping Template Matching - elapsed time: 2150 ms
- FAILED - score: 0.0 - Overlapping Template Matching - elapsed time: 1939 ms
- FAILED - score: 0.01 - Maurers Universal - elapsed time: 587 ms
- FAILED - score: 0.0 - Linear Complexity - elapsed time: 64999 ms
- FAILED - score: 0.0 - Serial - elapsed time: 23526 ms
- FAILED - score: 0.0 - Approximate Entropy - elapsed time: 15307 ms
- FAILED - score: 0.0 - Cumulative Sums - elapsed time: 625 ms
- FAILED - score: 0.683 - Random Excursion - elapsed time: 2045 ms
- PASSED - score: 0.17 - Random Excursion Variant - elapsed time: 89 ms

How seriously should I take this? Perhaps not *very*, because I just ran the same test on `binary_sequence = numpy.random.choice([0, 1], size=1200000)` and got similar results:

- PASSED - score: 0.64 - Monobit - elapsed time: 0 ms
- PASSED - score: 0.363 - Frequency Within Block - elapsed time: 1 ms
- PASSED - score: 0.214 - Runs - elapsed time: 228 ms
- PASSED - score: 0.63 - Longest Run Ones In A Block - elapsed time: 102 ms
- PASSED - score: 0.13 - Binary Matrix Rank - elapsed time: 1976 ms
- FAILED - score: 0.0 - Discrete Fourier Transform - elapsed time: 50 ms
- FAILED - score: 0.0 - Non Overlapping Template Matching - elapsed time: 2267 ms
- FAILED - score: 0.0 - Overlapping Template Matching - elapsed time: 1916 ms
- FAILED - score: 0.01 - Maurers Universal - elapsed time: 650 ms
- FAILED - score: 0.0 - Linear Complexity - elapsed time: 70173 ms
- FAILED - score: 0.0 - Serial - elapsed time: 26096 ms
- FAILED - score: 0.0 - Approximate Entropy - elapsed time: 16882 ms
- FAILED - score: 0.0 - Cumulative Sums - elapsed time: 673 ms
- FAILED - score: 0.683 - Random Excursion - elapsed time: 2190 ms
- PASSED - score: 0.17 - Random Excursion Variant - elapsed time: 11 ms

How about cryptographically secure bits? `import secrets; secrets.randbits(1200000)`

```python
import secrets

bits = secrets.randbits(1200000)
binary_sequence = numpy.array([int(i) for i in format(bits, 'b')], dtype=int)
```

- PASSED - score: 0.031 - Monobit - elapsed time: 0 ms
- PASSED - score: 0.31 - Frequency Within Block - elapsed time: 1 ms
- PASSED - score: 0.261 - Runs - elapsed time: 233 ms
- PASSED - score: 0.632 - Longest Run Ones In A Block - elapsed time: 102 ms
- PASSED - score: 0.05 - Binary Matrix Rank - elapsed time: 1986 ms
- FAILED - score: 0.0 - Discrete Fourier Transform - elapsed time: 48 ms
- PASSED - score: 0.621 - Non Overlapping Template Matching - elapsed time: 2346 ms
- FAILED - score: 0.0 - Overlapping Template Matching - elapsed time: 1942 ms
- PASSED - score: 0.01 - Maurers Universal - elapsed time: 643 ms
- FAILED - score: 0.0 - Linear Complexity - elapsed time: 71814 ms
- FAILED - score: 0.0 - Serial - elapsed time: 25466 ms
- FAILED - score: 0.0 - Approximate Entropy - elapsed time: 16863 ms
- FAILED - score: 0.0 - Cumulative Sums - elapsed time: 679 ms
- FAILED - score: 0.63 - Random Excursion - elapsed time: 2194 ms
- FAILED - score: 0.0 - Random Excursion Variant - elapsed time: 11 ms

Okay, the last test: `import os; os.urandom(1200000)`

```python
import os
bits = int.from_bytes(os.urandom(1200000), byteorder='big')
binary_sequence = numpy.array([int(i) for i in format(bits, 'b')], dtype=int)
```

- PASSED - score: 0.922 - Monobit - elapsed time: 3 ms
- PASSED - score: 0.536 - Frequency Within Block - elapsed time: 4 ms
- PASSED - score: 0.083 - Runs - elapsed time: 1840 ms
- PASSED - score: 0.685 - Longest Run Ones In A Block - elapsed time: 100 ms
- PASSED - score: 0.443 - Binary Matrix Rank - elapsed time: 15867 ms
- FAILED - score: 0.0 - Discrete Fourier Transform - elapsed time: 894 ms
- FAILED - score: 0.0 - Non Overlapping Template Matching - elapsed time: 18605 ms
- FAILED - score: 0.0 - Overlapping Template Matching - elapsed time: 1960 ms
- FAILED - score: 0.001 - Maurers Universal - elapsed time: 4573 ms
- FAILED - score: 0.0 - Linear Complexity - elapsed time: 561429 ms
- FAILED - score: 0.0 - Serial - elapsed time: 208261 ms
- FAILED - score: 0.0 - Approximate Entropy - elapsed time: 133384 ms
- FAILED - score: 0.0 - Cumulative Sums - elapsed time: 5544 ms
- FAILED - score: 0.63 - Random Excursion - elapsed time: 18154 ms
- FAILED - score: 0.0 - Random Excursion Variant - elapsed time: 71 ms

### Compressor arena

Trying to run the typical compressors on the binary file gives:

| Compression Tool                                    | Ratio                      |
| --------------------------------------------------- | -------------------------- |
| PAQ8PX (several settings)                           | 1.00                 |
| XZ                                                  | 0.99950973                 |
| ZSTD                                                | 0.99986091                 |
| Brotli                                              | 0.99996339                 |
| 7ZA                                                 | 0.99856713                 |

Even the best compressor (7ZA) is able to compress the file by only 0.14%. This is not surprising, as the file is already very close to random.

However this is much higher than what I think is theoretically possible. Theoretically, we should be able to compress Wikipedia to 0.8 bpc, and the file is already at 0.9 bpc, so we should be able to compress `wiki_arithmetic_code_gpt2-xl_test_1.bin` by a ratio of 0.8/0.9 = 89%.

## 02-21

Ran the NIST test suite again, this time in C. The only pseudorandomness test that failed is the Maurers Universal Test. This is bizarre because none of the compression algorithms I tried managed to compress the file by more than 0.14%. I'm not sure what to make of this.

I did check the test suite on `urandom` output, and that passed all tests. I have no idea what to do with this.

Actually I reimplemented the Maurers Universal Test in Python and ran it on `wiki_arithmetic_code_gpt2-xl_test_1.txt`. It passed. I'm not sure what to make of this.

Still, all evidence points to the probable conclusion that there is no "free lunch" in intelligence by route of compression. To compress an LLM-compressed file further, the second-stage compressor would have to be at least as powerful as the first-stage compressor. This can be intuitive, but disappointing.