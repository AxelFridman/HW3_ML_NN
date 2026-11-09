# HW3: MLP and CNN on CIFAR-10

Axel Fridman and Abdulhamit Gokce, Machine Learning & Neural Networks for Neuroscience, Bar-Ilan University.

| File | Task |
|---|---|
| [`1_mlp_vs_cnn.ipynb`](1_mlp_vs_cnn.ipynb) | MLP vs CNN: hyperparameter search, accuracy, size, speed, memory, weights, shuffled pixels |
| [`2_overfitting.ipynb`](2_overfitting.ipynb) | Small, base and large CNN: when overfitting begins; dropout vs weight decay on the FC layers |
| [`3_augmentation.ipynb`](3_augmentation.ipynb) | Crop, rotation, zoom and flips, each alone and combined; learning curves, confusion matrices |
| [`4_embeddings.ipynb`](4_embeddings.ipynb) | Frozen DINOv2-small embeddings + a small MLP, compared with the Task 1 models; t-SNE |
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
this is committed. A full run takes about 15 hours on a 4-core laptop CPU (`JOBS` in `hw3.py` sets the number of parallel
folds); with the cache in place, a notebook re-runs in a minute or two. Tested with Python 3.11, torch 2.7.1 (CPU),
torchvision 0.22.1 and transformers 4.52.4.
