"""Two-dimensional relief-style replacements for the former 3D figures.

The third figure shows the along-path intensity redistribution hidden by the
terminal energy law; the fifth shows the (xi_1, xi_4) quadratic risk surface
as a shaded contour map together with direct propagation. Both use the same
discretisation as scripts/study.py and the shared publication style.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from study import L, propagate, quantities, tangent, exact, setup
from plot_style import CMAP_SEQ, CMAP_DIV, apply, colorbar, shade_array

ROOT = Path(__file__).resolve().parents[1]


def propagate_snapshots(samples, epsilon, *, control=0., step=0.005, every=5,
                        size=1024, half_window=64., batch=64):
    """Propagate as in study.py but retain the field every `every` full steps."""
    samples = np.atleast_2d(samples)
    t, dt, k, _, left, right = setup(size, half_window, step, control)
    k2 = k * k
    chunks, frames = [], []
    for start in range(0, len(samples), batch):
        xi = epsilon * samples[start:start + batch]
        u = np.broadcast_to(exact(t, 0, control), (len(xi), size)).copy()
        local = [u.copy()]
        for index, (l, r) in enumerate(zip(left, right), 1):
            ib = l[0] + xi[:, 0] * l[2]
            ig = l[1] + xi[:, 2] * l[4] + xi[:, 3] * l[5]
            u = np.fft.ifft(np.fft.fft(u, axis=-1)
                            * np.exp(-0.5j * ib[:, None] * k2 + 0.5 * ig[:, None]), axis=-1)
            ing = l[0] + r[0] + xi[:, 1] * (l[3] + r[3])
            u *= np.exp(1j * ing[:, None] * np.abs(u) ** 2)
            ib = r[0] + xi[:, 0] * r[2]
            ig = r[1] + xi[:, 2] * r[4] + xi[:, 3] * r[5]
            u = np.fft.ifft(np.fft.fft(u, axis=-1)
                            * np.exp(-0.5j * ib[:, None] * k2 + 0.5 * ig[:, None]), axis=-1)
            if index % every == 0:
                local.append(u.copy())
        chunks.append(np.stack(local, axis=1))
        frames = len(local)
    z = np.arange(frames) * (L / round(L / step)) * every
    return t, z, np.concatenate(chunks, axis=0)


def map_axis(ax, *, extent, xlabel, ylabel, title, aspect='auto'):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect(aspect)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=8)
    ax.tick_params(direction='in', length=3.4, width=0.7, pad=3)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('0.35')


def build_propagation():
    apply()
    samples = np.array([[0., 0., 0., 0.], [0., 0., 0., 1.]])
    t, z, fields = propagate_snapshots(samples, 0.2, control=0.)
    keep = (t >= -16) & (t <= 16)
    tt = t[keep][::4]
    amp = np.abs(fields[:, :, keep][:, :, ::4])
    intensity = amp ** 2
    difference = intensity[1] - intensity[0]
    vdiff = float(np.max(np.abs(difference)))
    extent = [tt.min(), tt.max(), z.min(), z.max()]
    xx, yy = np.meshgrid(tt, z, indexing='xy')

    fig = plt.figure(figsize=(9.0, 3.8))
    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.15, top=0.88, wspace=0.30)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    norm1 = Normalize(0, float(intensity[0].max()))
    rgb1 = shade_array(intensity[0], CMAP_SEQ, norm1, fraction=1.15, vert_exag=1.0)
    ax1.imshow(rgb1, extent=extent, origin='lower', aspect='auto',
               interpolation='bicubic')
    ax1.contour(xx, yy, intensity[0], levels=8, colors='0.28',
                linewidths=0.42, alpha=0.55)
    map_axis(ax1, extent=extent, xlabel=r'$t$', ylabel=r'$z$',
             title='(a) nominal intensity relief')

    norm2 = Normalize(-vdiff, vdiff)
    base = plt.get_cmap(CMAP_DIV)(norm2(difference))
    ax2.imshow(base, extent=extent, origin='lower', aspect='auto',
               interpolation='bicubic')
    rgb2 = shade_array(difference, CMAP_DIV, norm2, fraction=0.9, vert_exag=0.8)
    rgb2[..., 3] = 0.55
    ax2.imshow(rgb2, extent=extent, origin='lower', aspect='auto',
               interpolation='bicubic')
    levels2 = np.array([-0.75, -0.5, -0.25, 0.25, 0.5, 0.75]) * vdiff
    ax2.contour(xx, yy, difference, levels=levels2, colors='0.25',
                linewidths=0.38, alpha=0.55)
    map_axis(ax2, extent=extent, xlabel=r'$t$', ylabel=r'$z$',
             title='(b) intensity change at fixed terminal energy')

    cax1 = fig.add_axes([0.862, 0.20, 0.014, 0.58])
    cax2 = fig.add_axes([0.942, 0.20, 0.014, 0.58])
    sm1 = ScalarMappable(norm=norm1, cmap=plt.get_cmap(CMAP_SEQ))
    sm2 = ScalarMappable(norm=norm2, cmap=plt.get_cmap(CMAP_DIV))
    colorbar(fig, sm1, cax1, r'$|u|^2$')
    colorbar(fig, sm2, cax2, r'$\Delta|u|^2$')
    fig.savefig(ROOT / 'figures' / 'fig5_propagation_2d.pdf', bbox_inches='tight')
    fig.savefig(ROOT / 'figures' / 'fig5_propagation_2d.png', bbox_inches='tight', dpi=260)
    plt.close(fig)
    return {'z': z, 't': tt, 'nominal': intensity[0].astype(np.float32),
            'difference': difference.astype(np.float32)}


def build_risk():
    apply()
    data = json.loads((ROOT / 'results' / 'study.json').read_text(encoding='utf-8'))
    H = np.array(data['responses']['0.0']['aligned_gram'])
    eps = 0.1
    grid = np.linspace(-1, 1, 25)
    x1, x4 = np.meshgrid(grid, grid, indexing='ij')
    xi = np.column_stack([x1.ravel(), np.zeros(x1.size),
                          np.zeros(x1.size), x4.ravel()])
    t, dt, k, u = propagate(xi, eps, control=0.)
    _, _, _, u_ref, _ = tangent(step=0.0025, control=0.)
    out = quantities(u, t, dt, k, xi, eps, u_ref)
    direct = out['aligned_loss'].reshape(x1.shape)
    quad = eps ** 2 * (H[0, 0] * x1 ** 2 + 2 * H[0, 3] * x1 * x4 + H[3, 3] * x4 ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(quad > 1e-14, direct / quad - 1., 0.)

    # Rows = xi_4 (y), columns = xi_1 (x), matching the plotted axes.
    direct_m, quad_m, rel_m = direct.T, quad.T, rel.T
    xx, yy = np.meshgrid(grid, grid, indexing='xy')
    extent = [grid.min(), grid.max(), grid.min(), grid.max()]

    fig = plt.figure(figsize=(9.0, 3.7))
    fig.subplots_adjust(left=0.07, right=0.84, bottom=0.16, top=0.88, wspace=0.30)
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    norm1 = Normalize(0, float(direct_m.max()))
    rgb1 = shade_array(direct_m, CMAP_SEQ, norm1, fraction=1.1, vert_exag=1.0)
    ax1.imshow(rgb1, extent=extent, origin='lower', aspect='equal',
               interpolation='bicubic')
    ax1.contour(xx, yy, quad_m, levels=7, colors='0.12', linewidths=0.62,
                linestyles='--', alpha=0.82)
    map_axis(ax1, extent=extent, xlabel=r'$\xi_1$', ylabel=r'$\xi_4$',
             title='(a) direct loss with quadratic contours', aspect='equal')
    ax1.text(0.03, 0.96, 'dashed: $\\epsilon^2Q$', transform=ax1.transAxes,
             fontsize=8.4, va='top',
             bbox=dict(facecolor='white', edgecolor='0.72', alpha=0.88, pad=2.2))

    vmax = float(max(0.02, np.max(np.abs(rel))))
    norm2 = Normalize(-vmax, vmax)
    rgb2 = shade_array(rel_m, CMAP_DIV, norm2, fraction=1.0, vert_exag=0.85)
    ax2.imshow(rgb2, extent=extent, origin='lower', aspect='equal',
               interpolation='bicubic')
    ax2.contour(xx, yy, rel_m, levels=9, colors='0.25',
                linewidths=0.38, alpha=0.45)
    ax2.contour(xx, yy, rel_m, levels=[0], colors='0.12', linewidths=0.85)
    map_axis(ax2, extent=extent, xlabel=r'$\xi_1$', ylabel=r'$\xi_4$',
             title='(b) relative higher-order departure', aspect='equal')

    cax1 = fig.add_axes([0.862, 0.20, 0.014, 0.58])
    cax2 = fig.add_axes([0.942, 0.20, 0.014, 0.58])
    sm1 = ScalarMappable(norm=norm1, cmap=plt.get_cmap(CMAP_SEQ))
    sm2 = ScalarMappable(norm=norm2, cmap=plt.get_cmap(CMAP_DIV))
    colorbar(fig, sm1, cax1, r'direct $1-F$')
    colorbar(fig, sm2, cax2, r'relative error')
    fig.savefig(ROOT / 'figures' / 'fig6_risk_2d.pdf', bbox_inches='tight')
    fig.savefig(ROOT / 'figures' / 'fig6_risk_2d.png', bbox_inches='tight', dpi=260)
    plt.close(fig)
    return {'grid': grid, 'direct': direct.astype(np.float32),
            'quadratic': quad.astype(np.float32), 'rel': rel.astype(np.float32)}


def main():
    propagation = build_propagation()
    risk = build_risk()
    np.savez_compressed(ROOT / 'results' / 'relief_data.npz',
                        **{'propagation_' + k: v for k, v in propagation.items()},
                        **{'risk_' + k: v for k, v in risk.items()})
    print('2D relief figures written; relative error range',
          float(risk['rel'].min()), float(risk['rel'].max()))


if __name__ == '__main__':
    main()
