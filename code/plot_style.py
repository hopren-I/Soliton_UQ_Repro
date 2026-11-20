"""Shared publication-style settings for all manuscript figures.

Colour semantics are fixed across the paper:
  sequential  -> non-negative magnitudes (intensity, infidelity)
  diverging   -> signed changes / errors
  cyclic      -> temporal phase
Geometric styling (line weights, panels, colourbars, lighting) is also shared.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, Normalize

BLUE = '#1f5b8f'
ORANGE = '#c15b2c'
TEAL = '#1c7a6b'
PURPLE = '#6d4c96'
GREY = '#4a4a4a'
LIGHT_GREY = '#b8b8b8'

CMAP_SEQ = 'viridis'
CMAP_DIV = 'RdBu_r'
CMAP_CYC = 'twilight'


def apply():
    """One typographic and line-style definition for every figure."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9.5,
        'axes.labelsize': 10,
        'axes.titlesize': 10.5,
        'legend.fontsize': 8.8,
        'xtick.labelsize': 8.6,
        'ytick.labelsize': 8.6,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.axisbelow': True,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'lines.linewidth': 1.7,
        'lines.markersize': 4.6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.minor.size': 2.0,
        'ytick.minor.size': 2.0,
    })


def finish_axis(ax, *, ygrid=True, xgrid=False):
    """Clean 2D panel with restrained grid lines and inward ticks."""
    if ygrid:
        ax.grid(axis='y', color='0.82', lw=0.6, alpha=0.75)
    if xgrid:
        ax.grid(axis='x', color='0.88', lw=0.5, alpha=0.6)
    ax.tick_params(direction='in', length=3.5, width=0.7, pad=3)
    for spine in ('left', 'bottom'):
        ax.spines[spine].set_linewidth(0.8)
        ax.spines[spine].set_color('0.35')
    ax.set_axisbelow(True)


def light():
    return LightSource(azdeg=315, altdeg=22)


def shade_array(values, cmap, norm=None, *, fraction=1.15, vert_exag=1.0):
    """Relief-style shading of a 2D scalar map, returned as RGBA for imshow."""
    cmap = plt.get_cmap(cmap)
    if norm is None:
        norm = Normalize(float(np.nanmin(values)), float(np.nanmax(values)))
    return light().shade(values, cmap=cmap, norm=norm, blend_mode='overlay',
                         fraction=fraction, vert_exag=vert_exag)


def colorbar(fig, mappable, cax, label, *, ticks=None, ticklabels=None):
    """Identical colourbar geometry, outline and typography everywhere."""
    cb = fig.colorbar(mappable, cax=cax)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor('0.62')
    cb.ax.tick_params(labelsize=8.4, length=2.6, width=0.6, pad=2)
    cb.set_label(label, labelpad=9, fontsize=9.5)
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    return cb
