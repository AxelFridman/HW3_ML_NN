import math
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.ticker import MaxNLocator
from torchvision.utils import make_grid

FIGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)

BLACK = '#0b0b0b'
DARK_GRAY = '#52514e'
GRAY = '#898781'
RED = '#e34948'
COLORS = {'mlp': '#eb6834', 'cnn': '#2a78d6', 'dino': '#6250d6', 'large': RED, 'dropout': '#008300', 'wd': '#eda100',
          'crop': '#eda100', 'novflip': '#008300', 'all': '#e87ba4'}

plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8.5, 'legend.fontsize': 7.5, 'lines.linewidth': 1.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': '#c3c2b7',
                     'xtick.color': DARK_GRAY, 'ytick.color': DARK_GRAY, 'axes.grid': True, 'grid.color': '#ebeae5',
                     'axes.axisbelow': True, 'legend.frameon': False, 'figure.dpi': 130, 'savefig.bbox': 'tight',
                     'figure.constrained_layout.use': True, 'pdf.fonttype': 42})


def save_fig(fig, name):
    fig.savefig(os.path.join(FIGS_DIR, f'{name}.pdf'))


def lighter(color, amount=0.5, towards='white'):
    return tuple((1 - amount) * a + amount * b for a, b in zip(to_rgb(color), to_rgb(towards)))


def plot_band(ax, runs, key, color, label=None, style='-', scale=1):
    # mean over the folds, with +-1 SD as a band
    v = np.stack([np.asarray(r[key], float) for r in runs]) * scale
    epochs = np.arange(1, v.shape[1] + 1)
    mean, sd = v.mean(0), v.std(0, ddof=1)
    ax.fill_between(epochs, mean - sd, mean + sd, color=color, alpha=0.2, lw=0)
    ax.plot(epochs, mean, style, color=color, label=label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def plot_curves(axes, groups):
    # groups: {label: (runs, color)}; left: loss, right: accuracy; dashed = train, solid = validation
    for label, (runs, color) in groups.items():
        plot_band(axes[0], runs, 'train_loss', color, None, '--')
        plot_band(axes[0], runs, 'val_loss', color, label, '-')
        plot_band(axes[1], runs, 'train_acc', color, None, '--', 100)
        plot_band(axes[1], runs, 'val_acc', color, label, '-', 100)
    axes[0].plot([], [], '--', color=GRAY, label='train')
    axes[0].plot([], [], '-', color=GRAY, label='validation')
    axes[0].set(xlabel='epoch', ylabel='cross-entropy loss')
    axes[1].set(xlabel='epoch', ylabel='accuracy (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].figure.legend(handles, labels, loc='outside upper center', ncol=len(groups) + 2)


def show_tiles(ax, x, nrow=8):
    # each tile rescaled from its min (dark) to its max (bright)
    x = x - x.amin((1, 2, 3), keepdim=True)
    x = x / x.amax((1, 2, 3), keepdim=True)
    ax.imshow(make_grid(x, nrow, 1, pad_value=1).permute(1, 2, 0), interpolation='nearest')
    ax.axis('off')


# ---- architecture drawings ----
# a layer is (shape, caption, color): shape is (channels, size) for a 3d block, a number for a flat vector,
# or a string for a plain rounded box

def cnn_layers(width=32, hidden=256):
    layers = []
    for i in range(3):
        channels, size = width * 2 ** i, 32 // 2 ** i
        layers.append(((channels, size), 'conv 3×3', COLORS['cnn']))
        layers.append(((channels, size // 2), 'max-pool', lighter(COLORS['cnn'])))
    layers.append((64 * width, 'flatten', GRAY))
    layers.append((hidden, 'FC + ReLU', COLORS['mlp']))
    layers.append((10, 'FC + softmax', BLACK))
    return layers


def draw_layers(ax, layers, img=None):
    right, pad = 0, 0
    if img is not None:
        ax.imshow(img.permute(1, 2, 0), extent=(0, 1.2, -0.6, 0.6))
        ax.text(0.6, -0.68, '32×32×3', ha='center', va='top', fontsize=6.5, color=DARK_GRAY)
        right = 1.2
    for shape, caption, color in layers:
        if isinstance(shape, str):
            # a plain box, as wide as its longest line of text
            longest = max(len(line) for line in shape.split('\n'))
            w = 0.25 + 0.11 * longest
            h = 0.8
            d = 0
            dims = ''
        elif isinstance(shape, tuple):
            # a 3d block for (channels, size)
            n, s = shape
            w = 0.055 * n ** 0.5
            h = 0.26 * s ** 0.5
            d = 0.078 * s ** 0.5
            dims = f'{s}×{s}×{n}'
        else:
            # a thin bar for a vector of `shape` numbers
            w = 0.1
            h = 0.17 * math.log2(shape)
            d = 0.06
            dims = f'{shape}'
        x = right + pad + 0.36
        pad = max(0, 0.105 * max(len(caption), len(dims)) - w - d) / 2  # room for long captions
        x += pad
        if right:
            ax.annotate('', (x - 0.05, 0), (right + 0.05, 0), arrowprops=dict(arrowstyle='-|>', color=GRAY, lw=0.8))
        if isinstance(shape, str):
            ax.add_patch(FancyBboxPatch((x, -h / 2), w, h, 'round,pad=0,rounding_size=.08', fc=color, ec='none'))
            ax.text(x + w / 2, 0, shape, ha='center', va='center', fontsize=7, color='white')
        else:
            front = [(0, 0), (w, 0), (w, h), (0, h)]
            top = [(0, h), (w, h), (w + d, h + d), (d, h + d)]
            side = [(w, 0), (w + d, d), (w + d, h + d), (w, h)]
            for face, shade in ((front, color), (top, lighter(color, 0.35)), (side, lighter(color, 0.25, 'black'))):
                ax.add_patch(Polygon([(x + a, b - h / 2) for a, b in face], fc=shade, ec='white', lw=0.5))
        ax.text(x + w / 2, -h / 2 - 0.08, dims, ha='center', va='top', fontsize=6.5, color=DARK_GRAY)
        ax.text(x + (w + d) / 2, h / 2 + d + 0.06, caption, ha='center', va='bottom', fontsize=7, color=BLACK)
        right = x + w + d
    ax.set(ylim=(-1.35, 1.5), aspect='equal')
    ax.axis('off')
    return right + pad + 0.1


def draw_architecture(name, rows, img=None):
    fig, axes = plt.subplots(len(rows), 1, figsize=(7.2, 1.5 * len(rows)), squeeze=False)
    end = 0
    for ax, layers in zip(axes[:, 0], rows):
        end = max(end, draw_layers(ax, layers, img))
    for ax in axes[:, 0]:
        ax.set_xlim(-0.05, end)  # same scale in every row
    save_fig(fig, name)
