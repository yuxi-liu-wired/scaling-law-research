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

