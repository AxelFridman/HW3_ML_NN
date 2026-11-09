# HW3: MLP, CNN and hybrids on CIFAR-10

Axel Fridman and Abdulhamit Gokce, Machine Learning & Neural Networks for Neuroscience, Bar-Ilan University.

| File | Task |
|---|---|
| [`1_mlp_vs_cnn.ipynb`](1_mlp_vs_cnn.ipynb) | MLP vs CNN: hyperparameter search, accuracy, size, speed, memory, weights, shuffled pixels |
| [`2_overfitting.ipynb`](2_overfitting.ipynb) | A larger CNN overfits: onset by a generalization-loss threshold; dropout vs weight decay |
| [`3_augmentation.ipynb`](3_augmentation.ipynb) | Crop, rotation, zoom and flips, each alone and combined; learning curves, confusion matrices |
| [`4_hybrids.ipynb`](4_hybrids.ipynb) | CNN features + one self-attention layer + MLP; frozen DINOv2-small embeddings + MLP |
| [`hw3.py`](hw3.py) | Data, folds, models, training loop, plots |

Every experiment uses 5-fold cross-validation on the 50,000 training images (folds of 10,000): each fold's model trains on
the other 40,000 images, is validated on its fold and is tested on the 10,000 test images. Numbers are mean (SD) over the
5 folds; the seed is 18042001.

## Run

```
pip install -r requirements.txt
```

Then run the notebooks in order: Task 1 picks the MLP and the CNN that the other tasks reuse. CIFAR-10 downloads to `data/`
and DINOv2-small comes from Hugging Face. Trained runs are cached in `out/runs/` and figures are saved to `out/figs/`; none of
this is committed. A full run takes about TOTAL hours on a 4-core laptop CPU (`JOBS` in `hw3.py` sets the number of parallel
folds); with the cache in place, a notebook re-runs in a minute or two. Tested with Python 3.11, torch 2.7.1 (CPU),
torchvision 0.22.1 and transformers 4.52.4.
