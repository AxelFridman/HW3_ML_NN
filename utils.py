import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets import CIFAR10

SEED = 18042001
N_FOLDS = 5
BATCH_SIZE = 128
N_JOBS = 4  # folds trained at the same time, one CPU core each
os.environ['LOKY_MAX_CPU_COUNT'] = str(N_JOBS)  # otherwise joblib warns on windows that it can't count the cores

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
OUT_DIR = os.path.join(ROOT, 'out')
os.makedirs(os.path.join(OUT_DIR, 'results'), exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, 'models'), exist_ok=True)

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
AUGMENTATIONS = ('crop', 'rotation', 'zoom', 'h-flip', 'v-flip')

# one fixed random order of the 32x32 pixel positions, for the shuffled-pixels experiment
PIXEL_ORDER = torch.randperm(1024, generator=torch.Generator().manual_seed(SEED))


def load_cifar():
    train = CIFAR10(DATA_DIR, train=True, download=True)
    test = CIFAR10(DATA_DIR, train=False, download=True)
    X_train = torch.tensor(train.data).permute(0, 3, 1, 2).contiguous()  # (N, 3, 32, 32), uint8
    X_test = torch.tensor(test.data).permute(0, 3, 1, 2).contiguous()
    return X_train, torch.tensor(train.targets), X_test, torch.tensor(test.targets)


def make_folds(y):
    # 5 stratified folds of 10,000 images, as (train indices, validation indices)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    folds = []
    for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
        folds.append((torch.from_numpy(train_idx), torch.from_numpy(val_idx)))
    return folds


def shuffle_pixels(x):
    # x is (N, 3, 32, 32): every image and every channel gets the same new order of its 1024 pixels
    n = len(x)
    x = x.reshape(n, 3, 1024)
    x = x[:, :, PIXEL_ORDER]
    return x.reshape(n, 3, 32, 32)


def load_data(data='pixels'):
    X_train, y_train, X_test, y_test = load_cifar()
    if data == 'shuffled':
        X_train, X_test = shuffle_pixels(X_train), shuffle_pixels(X_test)
    if data == 'dino':  # DINOv2 embeddings, computed once in notebook 4
        emb = torch.load(os.path.join(DATA_DIR, 'dino.pt'))
        X_train, X_test = emb['train'], emb['test']
    return X_train, y_train, X_test, y_test


def prepare(x, data='pixels'):
    # images -> floats with mean 0 and std 1 per channel (the embeddings are used as they are)
    if data == 'dino':
        return x
    x = (x / 255 - MEAN) / STD
    # channels_last makes the CNN almost 2x faster on our CPU
    return x.contiguous(memory_format=torch.channels_last)


def augment(x, ops, fixed=False):
    # random crop (shift), rotation, zoom and flips, all done with one affine warp per image.
    # fixed=True uses the largest setting of each one (for the pictures in the report)
    n = len(x)
    if fixed:
        angle = torch.full((n,), 15.0)
        zoom = torch.full((n,), 0.3)
    else:
        angle = torch.empty(n).uniform_(-15, 15)
        zoom = torch.empty(n).uniform_(0, 0.3)
    hflip = torch.rand(n) < 0.5
    vflip = torch.rand(n) < 0.5
    if fixed:
        hflip[:] = True
        vflip[:] = True
        shift = torch.full((n, 2), 4)
    else:
        shift = torch.randint(-4, 5, (n, 2))

    # turn off what is not in ops
    if 'rotation' not in ops:
        angle = torch.zeros(n)
    if 'zoom' not in ops:
        zoom = torch.zeros(n)
    if 'h-flip' not in ops:
        hflip[:] = False
    if 'v-flip' not in ops:
        vflip[:] = False
    if 'crop' not in ops:
        shift = torch.zeros(n, 2)

    angle = torch.deg2rad(angle)
    scale = 1 - zoom
    sx = torch.where(hflip, -1.0, 1.0)
    sy = torch.where(vflip, -1.0, 1.0)
    shift = shift * 2 / 32  # affine_grid wants the shift in [-1, 1] coordinates, so 4 px = 0.25

    # the 2x3 matrix of each image
    cos = scale * angle.cos()
    sin = scale * angle.sin()
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = cos * sx
    theta[:, 0, 1] = -sin * sy
    theta[:, 0, 2] = shift[:, 0]
    theta[:, 1, 0] = sin * sx
    theta[:, 1, 1] = cos * sy
    theta[:, 1, 2] = shift[:, 1]
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x.float(), grid, align_corners=False)


def evaluate(model, X, y, data='pixels'):
    model.eval()
    outputs = []
    with torch.no_grad():
        for xb in X.split(1000):
            outputs.append(model(prepare(xb, data)))
    outputs = torch.cat(outputs)
    loss = F.cross_entropy(outputs, y).item()
    preds = outputs.argmax(1)
    acc = (preds == y).float().mean().item()
    return loss, acc, preds


def results_path(name, fold):
    return os.path.join(OUT_DIR, 'results', f'{name}_fold{fold}.pt')


def model_path(name, fold):
    return os.path.join(OUT_DIR, 'models', f'{name}_fold{fold}.pt')


def train_fold(name, fold, model_class, model_args, lr, epochs=10, weight_decay=0.0, augmentations=(), data='pixels'):
    torch.set_num_threads(1)
    torch.manual_seed(SEED + fold)
    X, y, X_test, y_test = load_data(data)
    train_idx, val_idx = make_folds(y)[fold]
    model = model_class(**model_args).to(memory_format=torch.channels_last)  # same memory layout as the images

    # weight decay only on the fully connected layers
    fc_params, other_params = [], []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            fc_params += list(module.parameters())
        elif isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            other_params += list(module.parameters())
    optimizer = torch.optim.AdamW([{'params': fc_params, 'weight_decay': weight_decay},
                                   {'params': other_params, 'weight_decay': 0.0}], lr=lr)
    # learning rate: warm-up to lr in the first 30% of the steps, then decay to almost 0
    steps = epochs * math.ceil(len(train_idx) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'epoch_time': []}
    for epoch in range(epochs):
        model.train()
        start = time.perf_counter()
        shuffled = train_idx[torch.randperm(len(train_idx))]
        for batch in shuffled.split(BATCH_SIZE):
            xb = X[batch]
            if augmentations:
                xb = augment(xb, augmentations)
            loss = F.cross_entropy(model(prepare(xb, data)), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        history['epoch_time'].append(time.perf_counter() - start)

        # train metrics on a quarter of the training images, without augmentation
        train_loss, train_acc, _ = evaluate(model, X[train_idx[::4]], y[train_idx[::4]], data)
        val_loss, val_acc, _ = evaluate(model, X[val_idx], y[val_idx], data)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

    test_loss, test_acc, test_pred = evaluate(model, X_test, y_test, data)
    history['test_loss'] = test_loss
    history['test_acc'] = test_acc
    history['test_pred'] = test_pred.byte()
    history['n_params'] = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), model_path(name, fold))  # save the model just in case
    torch.save(history, results_path(name, fold))


def load_results(name):
    # what train_fold saved: a list with one dict per fold
    return [torch.load(results_path(name, fold)) for fold in range(N_FOLDS)]


def load_model(name, model_class, model_args, fold=0):
    model = model_class(**model_args)
    model.load_state_dict(torch.load(model_path(name, fold)))
    return model.eval()


def benchmark(model, X, y, data='pixels', steps=30):
    # seconds per epoch (4 threads) and training memory in MB, measured on one batch of 128 images
    torch.set_num_threads(4)
    x = prepare(X[:BATCH_SIZE], data)
    labels = y[:BATCH_SIZE]
    model = model.to(memory_format=torch.channels_last)

    # memory: the weights, their gradients and the 2 Adam moments, plus what autograd keeps for the backward pass
    # (saved_tensors_hooks, from the pytorch docs on autograd hooks: keep() sees every tensor saved for backward)
    saved = {}
    params = {p.data_ptr() for p in model.parameters()}

    def keep(t):
        if t.data_ptr() not in params:
            saved[t.data_ptr()] = t.nbytes
        return t

    with torch.autograd.graph.saved_tensors_hooks(keep, lambda t: t):
        model(x)
    n = sum(p.numel() for p in model.parameters())

    # time: the fastest of 30 training steps (the laptop is also doing other things)
    optimizer = torch.optim.AdamW(model.parameters())
    times = []
    for step in range(steps + 3):  # 3 extra warm-up steps at the start, not counted (times[3:] below)
        start = time.perf_counter()
        optimizer.zero_grad()
        F.cross_entropy(model(x), labels).backward()
        optimizer.step()
        times.append(time.perf_counter() - start)
    steps_per_epoch = len(X) * (N_FOLDS - 1) / N_FOLDS / BATCH_SIZE

    mb = 2 ** 20
    result = {}
    result['params'] = n
    result['weights_mb'] = 4 * n / mb  # float32: 4 bytes per weight
    result['adam_mb'] = 12 * n / mb  # gradient + 2 Adam moments, 4 bytes each
    result['activations_mb'] = sum(saved.values()) / mb
    result['epoch_seconds'] = min(times[3:]) * steps_per_epoch
    return result


def last(runs, key):
    # one number per fold: the last epoch for the lists (val_acc, ...), the number itself for test_acc
    values = []
    for r in runs:
        if isinstance(r[key], list):
            values.append(r[key][-1])
        else:
            values.append(r[key])
    return np.array(values, dtype=float)


def mean_sd(values, scale=100, decimals=1):
    v = np.array(values, dtype=float) * scale
    return f'{v.mean():.{decimals}f} ({v.std(ddof=1):.{decimals}f})'


def difference(runs_a, runs_b, key='test_acc'):
    # a minus b in each fold, and in how many folds a is better
    d = last(runs_a, key) - last(runs_b, key)
    text = mean_sd(d)
    if d.mean() >= 0:
        text = '+' + text
    return text + ', ' + str((d > 0).sum()) + '/' + str(len(d)) + ' folds'


def latex(v):
    # real minus sign for negative numbers like -1.2 (not for 1024-512-256, h-flip or the '--' in the table)
    v = str(v)
    if v.startswith('-') and v != '--':
        v = '$-$' + v[1:]
    return v


def save_numbers(name, numbers):
    # numbers for the report, as LaTeX commands
    with open(os.path.join(OUT_DIR, f'{name}.tex'), 'w') as f:
        for key, value in numbers.items():
            f.write('\\newcommand{\\' + key + '}{' + latex(value) + '}\n')


def save_table(name, df):
    with open(os.path.join(OUT_DIR, f'{name}.tex'), 'w') as f:
        for row in df.itertuples():
            f.write(' & '.join(latex(v) for v in row) + ' \\\\\n')
