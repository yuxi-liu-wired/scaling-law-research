# Readme

## Datasets

The datasets are from the [wikitext-103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) dataset. The dataset is a collection of over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. The dataset is available under the Creative Commons Attribution-ShareAlike License.

For processing into the Shannon guessing game format, each dataset was processed by deleting every character that is not one of `abcdefghijklmnopqrstuvwxyz .,`, and then all whitespace was replaced with a single space. The resulting character counts of each of the datasets are as follows.

```txt
   1288556 wiki.test.raw
 539566975 wiki.train.raw
   1144748 wiki.valid.raw
   1215381 wiki_shannonfied.test.txt
 508906728 wiki_shannonfied.train.txt
   1080524 wiki_shannonfied.valid.txt
1053203781 total
```

The largest dataset has 5e8 chars, and the smallest has 1e6 chars. For the HMM experiment we only use up to 2e7 chars in the `wiki.train.raw`.

## HMM

To continue from where we left off,

```python
import pickle
with open("filename.pkl", "wb") as file:
    pickle.dump(model, file)
with open("filename.pkl", "rb") as file:
    model = pickle.load(file)
model.init_params = ''
```

## Grid search sweep

The grid search sweep was performed on the `wiki.train.raw` dataset. The grid search was performed on the following parameters:

* Dataset size ($D$): 2e4 -- 2e7;
* `n_components`: 3 -- 300;
* Model parameters ($N$): 1e2 -- 1e5;

```python
train_sizes = [i*10**4 for i in [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]]
n_components_list = [3, 5, 9, 16, 27, 47, 75, 110, 156, 230, 300]
```

Each configuration was run for 5 times. For SLURM, they recommend `--count 1`

```bash
wandb agent --count 5 <sweep_id>
```

Went forth and let it run overnight. Took way too long. Will try something smaller.

I found a scaling law for the time it takes to run one EM step:

$$
Time/\;\mathrm{hour} = \frac{N^{1.7}D}{(3524)^{1.7}(20\;\mathrm{million})}
$$

According to this law, I cannot afford more than $N = 28k$.