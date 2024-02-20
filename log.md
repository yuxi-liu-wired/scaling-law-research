# Research log

## 02-20

Got the arithmetic compression working, on about 1.2 million characters (`wiki103.test`).

| Model       | Compressed Size (bits) | Compressed Size (MB) | Arithmetic Interval Length (bits) | Bit Rate (bit/token) | Bit Rate (bit/char) |
| ----------- | ---------------------- | -------------------- | --------------------------------- | -------------------- | ------------------- |
| gpt2        | 1,334,322              | 0.16                 | 1,334,328.88                      | 5.18                 | 1.10                |
| gpt2-medium | 1,187,137              | 0.14                 | 1,187,136.62                      | 4.61                 | 0.98                |
| gpt2-large  | 1,133,520              | 0.14                 | 1,133,512.62                      | 4.40                 | 0.93                |
| gpt2-xl     | 1,092,744              | 0.13                 | 1,092,743.00                      | 4.24                 | 0.90                |

