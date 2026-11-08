import json
import math
import os
import time
from functools import cache, partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Polygon
from sklearn.model_selection import StratifiedKFold
from torch.utils.flop_counter import FlopCounterMode
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid

SEED, FOLDS, BATCH, JOBS = 18042001, 5, 128, 4
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(ROOT, 'data'), os.path.join(ROOT, 'out')
FIGS, RUNS = os.path.join(OUT, 'figs'), os.path.join(OUT, 'runs')
os.makedirs(FIGS, exist_ok=True)
CLASSES = 'airplane automobile bird cat deer dog frog horse ship truck'.split()
MEAN = torch.tensor([.4914, .4822, .4465]).view(1, 3, 1, 1)
STD = torch.tensor([.2470, .2435, .2616]).view(1, 3, 1, 1)
PERM = torch.randperm(1024, generator=torch.Generator().manual_seed(SEED))
AUG = {'all': ('crop', 'rotation', 'zoom', 'h-flip', 'v-flip'), 'novflip': ('crop', 'rotation', 'zoom', 'h-flip')}

INK, TEXT, MUTED = '#0b0b0b', '#52514e', '#898781'
COL = dict(mlp='#eb6834', cnn='#2a78d6', attn='#1baf7a', dino='#6250d6', large='#e34948', dropout='#008300',
           wd='#eda100', novflip='#008300', all='#e87ba4')
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'legend.fontsize': 9, 'lines.linewidth': 1.8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#c3c2b7',
                     'xtick.color': TEXT, 'ytick.color': TEXT, 'axes.grid': True, 'grid.color': '#ebeae5',
                     'axes.axisbelow': True, 'legend.frameon': False, 'figure.dpi': 110, 'savefig.bbox': 'tight',
                     'pdf.fonttype': 42})


@cache
def cifar():
    sets = [CIFAR10(DATA, train, download=True) for train in (True, False)]
    return [t for s in sets for t in (torch.tensor(s.data).permute(0, 3, 1, 2).contiguous(), torch.tensor(s.targets))]


@cache
def folds():
    y = cifar()[1]
    return [torch.from_numpy(i) for _, i in StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(y, y)]


@cache
def inputs(kind):
    X, Y, Xte, Yte = cifar()
    if kind == 'dino':
        Z = torch.load(os.path.join(DATA, 'dino.pt'))
        return Z['train'], Y, Z['test'], Yte
    return X, Y, Xte, Yte


def augment(x, ops, p=.5):
    n, u = len(x), lambda a, b: torch.empty(len(x)).uniform_(a, b)
    a = torch.deg2rad(u(-15, 15)) * ('rotation' in ops)
    s = u(.7, 1) if 'zoom' in ops else torch.ones(n)
    fx, fy = (torch.where((torch.rand(n) < p) & (k in ops), -1., 1.) for k in ('h-flip', 'v-flip'))
    t = torch.randint(-4, 5, (n, 2)) / 16 * ('crop' in ops)
    theta = torch.stack([s * a.cos() * fx, -s * a.sin() * fy, t[:, 0], s * a.sin() * fx, s * a.cos() * fy, t[:, 1]], 1)
    return F.grid_sample(x.float(), F.affine_grid(theta.view(n, 2, 3), x.shape, False), align_corners=False)


def prep(x, kind='pixels'):
    if kind == 'dino':
        return x
    if kind == 'perm':
        x = x.flatten(2)[..., PERM].view_as(x)
    return ((x / 255 - MEAN) / STD).contiguous(memory_format=torch.channels_last)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))


def MLP(hidden=(512, 256), p=0., d=3072):
    h = (d, *hidden)
    return nn.Sequential(nn.Flatten(), *[m for a, b in zip(h, h[1:]) for m in (nn.Linear(a, b), nn.ReLU(), nn.Dropout(p))],
                         nn.Linear(h[-1], 10))


def CNN(w=32, fc=256, p=0.):
    return nn.Sequential(block(3, w), block(w, 2 * w), block(2 * w, 4 * w), nn.Flatten(), nn.Dropout(p),
                         nn.Linear(64 * w, fc), nn.ReLU(), nn.Dropout(p), nn.Linear(fc, 10))


class AttnCNN(nn.Module):
    def __init__(self, w=32, heads=4, fc=256):
        super().__init__()
        d = 4 * w
        self.cnn = nn.Sequential(block(3, w), block(w, 2 * w), *block(2 * w, d)[:3])
        self.cls, self.pos = nn.Parameter(torch.zeros(1, 1, d)), nn.Parameter(torch.randn(1, 65, d) / 50)
        self.norm, self.attn = nn.LayerNorm(d), nn.MultiheadAttention(d, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, fc), nn.GELU(), nn.Linear(fc, 10))

    def forward(self, x, maps=False):
        t = self.cnn(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls.expand(len(t), -1, -1), t], 1) + self.pos
        n = self.norm(t)
        a, w = self.attn(n[:, :1], n, n, need_weights=maps)
        y = self.mlp(t[:, 0] + a[:, 0])
        return (y, w[:, 0, 1:]) if maps else y


@torch.no_grad()
def evaluate(net, X, Y, kind='pixels'):
    net.eval()
    out = torch.cat([net(prep(x, kind)) for x in X.split(1000)])
    return F.cross_entropy(out, Y).item(), (out.argmax(1) == Y).float().mean().item(), out.argmax(1)


def wnorm(net):
    return math.sqrt(sum(p.detach().square().sum().item() for p in net.parameters() if p.dim() > 1))


def train(name, fold, model, epochs=10, lr=1e-3, wd=0., aug=None, kind='pixels'):
    torch.set_num_threads(1)
    torch.manual_seed(SEED + fold)
    X, Y, Xte, Yte = inputs(kind)
    va = folds()[fold]
    tr = torch.from_numpy(np.setdiff1d(np.arange(len(Y)), va))
    net = model().to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * math.ceil(len(tr) / BATCH))
    log = []
    for _ in range(epochs):
        net.train()
        t, s = time.perf_counter(), np.zeros(3)
        for i in tr[torch.randperm(len(tr))].split(BATCH):
            x, y = X[i], Y[i]
            if aug:
                x = augment(x, AUG[aug])
            out = net(prep(x, kind))
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            s += loss.item() * len(y), (out.argmax(1) == y).sum().item(), len(y)
        t = time.perf_counter() - t
        log.append([s[0] / s[2], s[1] / s[2], *evaluate(net, X[va], Y[va], kind)[:2], t, wnorm(net)])
    loss, acc, pred = evaluate(net, Xte, Yte, kind)
    res = dict(zip(('train_loss', 'train_acc', 'val_loss', 'val_acc', 'time', 'wnorm'), torch.tensor(log).T))
    os.makedirs(d := os.path.join(RUNS, name), exist_ok=True)
    torch.save(net.state_dict(), os.path.join(d, f'{fold}.pth'))
    torch.save(res | dict(test_loss=loss, test_acc=acc, pred=pred.byte(), params=sum(p.numel() for p in net.parameters())),
               os.path.join(d, f'{fold}.pt'))


def cv(specs):
    path = lambda k, f: os.path.join(RUNS, k, f'{f}.pt')
    todo = [(k, f) for k in specs for f in range(FOLDS) if not os.path.exists(path(k, f))]
    Parallel(JOBS)(delayed(train)(k, f, **specs[k]) for k, f in todo)
    return {k: [torch.load(path(k, f)) for f in range(FOLDS)] for k in specs}


def load(name, model, fold=0):
    net = model()
    net.load_state_dict(torch.load(os.path.join(RUNS, name, f'{fold}.pth')))
    return net.eval()


def profile(model, kind='pixels', steps=20):
    torch.set_num_threads(4)
    X, Y, *_ = inputs(kind)
    x, y, acts = prep(X[:BATCH], kind), Y[:BATCH], []
    net = model().to(memory_format=torch.channels_last)
    hooks = [m.register_forward_hook(lambda m, i, o: acts.append(o.numel())) for m in net.modules() if not [*m.children()]]
    with FlopCounterMode(display=False) as flops:
        net(x)
    for h in hooks:
        h.remove()
    opt = torch.optim.AdamW(net.parameters())
    for i in range(steps + 3):
        if i == 3:
            t = time.perf_counter()
        opt.zero_grad()
        F.cross_entropy(net(x), y).backward()
        opt.step()
    n = sum(p.numel() for p in net.parameters())
    return dict(params=n, mflops=flops.get_total_flops() / BATCH / 1e6, epoch=(time.perf_counter() - t) / steps * 40000 / BATCH,
                weights=4 * n / 2 ** 20, adam=12 * n / 2 ** 20, acts=4 * sum(acts) / 2 ** 20)


def col(runs, key):
    return np.array([np.ravel(r[key])[-1] for r in runs], float)


def ms(v, k=100, d=1):
    v = np.asarray(v, float) * k
    return f'{v.mean():.{d}f} ({v.std(ddof=1):.{d}f})'


def mix(c, t='white', a=.5):
    return tuple((1 - a) * u + a * v for u, v in zip(to_rgb(c), to_rgb(t)))


def band(ax, runs, key, color, label=None, ls='-', k=1):
    v = np.stack([np.asarray(r[key], float) for r in runs]) * k
    e, m, s = np.arange(1, v.shape[1] + 1), v.mean(0), v.std(0, ddof=1)
    ax.fill_between(e, m - s, m + s, color=color, alpha=.2, lw=0)
    ax.plot(e, m, ls, color=color, label=label)


def curves(groups, axes=None):
    axes = plt.subplots(1, 2, figsize=(9, 3.2))[1] if axes is None else axes
    for label, (runs, color) in groups.items():
        for ax, m, k in zip(axes, ('loss', 'acc'), (1, 100)):
            band(ax, runs, f'train_{m}', color, None, '--', k)
            band(ax, runs, f'val_{m}', color, label, '-', k)
    for ls, label in (('--', 'train'), ('-', 'validation')):
        axes[0].plot([], [], ls, color=MUTED, label=label)
    axes[0].set(xlabel='epoch', ylabel='cross-entropy loss')
    axes[1].set(xlabel='epoch', ylabel='accuracy (%)')
    axes[0].legend()
    return axes


def tiles(ax, x, nrow=8):
    x = x - x.amin((1, 2, 3), keepdim=True)
    ax.imshow(make_grid(x / x.amax((1, 2, 3), keepdim=True), nrow, 1, pad_value=1).permute(1, 2, 0), interpolation='nearest')
    ax.axis('off')


def cnn_layers(w=32, fc=256):
    return [((w, 32), 'conv 3×3', COL['cnn']), ((2 * w, 16), 'conv 3×3', COL['cnn']), ((4 * w, 8), 'conv 3×3', COL['cnn']),
            ((4 * w, 4), 'max-pool', mix(COL['cnn'])), (64 * w, 'flatten', MUTED), (fc, 'FC + ReLU', COL['mlp']),
            (10, 'softmax', INK)]


def draw(layers, img=None, ax=None):
    ax = ax or plt.subplots(figsize=(11, 2.4))[1]
    x = 0
    if img is not None:
        ax.imshow(img.permute(1, 2, 0), extent=(0, 1.2, -.6, .6))
        ax.text(.6, -.68, '32×32×3', ha='center', va='top', fontsize=7.5, color=TEXT)
        x = 1.2
    for shape, cap, c in layers:
        ax.annotate('', (x + .42, 0), (x + .08, 0), arrowprops=dict(arrowstyle='-|>', color=MUTED, lw=.8))
        x += .5
        if isinstance(shape, str):
            w, h, d, dims = .2 + .088 * max(map(len, shape.split('\n'))), .8, 0, ''
            ax.add_patch(FancyBboxPatch((x, -h / 2), w, h, 'round,pad=0,rounding_size=.08', fc=c, ec='none'))
            ax.text(x + w / 2, 0, shape, ha='center', va='center', fontsize=8, color='white')
        else:
            n, s = shape if isinstance(shape, tuple) else (shape, 0)
            w, h, d = (.055 * n ** .5, .26 * s ** .5, .078 * s ** .5) if s else (.1, .17 * math.log2(n), .06)
            dims = f'{s}×{s}×{n}' if s else f'{n}'
            for f, k in (([(0, 0), (w, 0), (w, h), (0, h)], 0), ([(0, h), (w, h), (w + d, h + d), (d, h + d)], .35),
                         ([(w, 0), (w + d, d), (w + d, h + d), (w, h)], -.25)):
                ax.add_patch(Polygon([(x + a, b - h / 2) for a, b in f], fc=mix(c, 'white' if k > 0 else 'black', abs(k)),
                                     ec='white', lw=.5))
        ax.text(x + w / 2, -h / 2 - .08, dims, ha='center', va='top', fontsize=7.5, color=TEXT)
        ax.text(x + (w + d) / 2, h / 2 + d + .06, cap, ha='center', va='bottom', fontsize=8, color=INK)
        x += w + d
    ax.set(xlim=(-.05, x + .1), ylim=(-1.35, 1.5), aspect='equal')
    ax.axis('off')
    return ax


def save(fig, name):
    fig.savefig(os.path.join(FIGS, f'{name}.pdf'))


def tex(name, **kv):
    with open(os.path.join(OUT, f'{name}.tex'), 'w') as f:
        f.writelines(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in kv.items())
