"""Four 2D figures built only from saved study outputs, in the shared style."""
from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from plot_style import (BLUE, ORANGE, TEAL, PURPLE, GREY, CMAP_DIV,
                        apply, finish_axis, colorbar)

ROOT = Path(__file__).resolve().parents[1]


def build(res):
    from study import profile, propagate, exact, L
    apply()
    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)

    def save(fig, name):
        fig.savefig(out / (name + '.pdf'), bbox_inches='tight')
        fig.savefig(out / (name + '.png'), bbox_inches='tight', dpi=260)
        plt.close(fig)

    def clean(ax):
        finish_axis(ax)
        ax.xaxis.set_major_locator(MaxNLocator(5))

    best = res['design']['best']['control']
    z = np.linspace(0, L, 601)

    # Fig. 1: prescribed family and error modes.
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 4.5), layout='constrained')
    for a, color, label in [(0., BLUE, r'$a=0$'), (best, ORANGE, rf'$a={best:g}$')]:
        b, w = profile(z, a)
        axes[0, 0].plot(z, b, color=color, label=label)
        axes[0, 1].plot(z, -b / w, color=color, label=label)
    axes[0, 0].set(title='(a) Dispersion and nonlinearity',
                   xlabel=r'$z$', ylabel=r'$\beta_0=\gamma_0$')
    axes[0, 0].legend(frameon=False, ncol=2, loc='upper right')
    axes[0, 1].set(title='(b) Nominal gain / loss', xlabel=r'$z$', ylabel=r'$g_0$')
    axes[0, 1].legend(frameon=False, ncol=2, loc='lower right')
    axes[1, 0].plot(z, np.sin(np.pi * z / L) ** 2, color=BLUE, label=r'$f_\beta$')
    axes[1, 0].plot(z, np.sin(2 * np.pi * z / L) ** 2, color=ORANGE, label=r'$f_\gamma$')
    axes[1, 0].set(title='(c) Relative coefficient error modes',
                   xlabel=r'$z$', ylabel='mode amplitude')
    axes[1, 0].legend(frameon=False, ncol=2, loc='lower center')
    axes[1, 1].plot(z, z / L, color=TEAL, label=r'$\int_0^z h_0\,ds$')
    axes[1, 1].plot(z, np.sin(2 * np.pi * z / L), color=PURPLE,
                    label=r'$\int_0^z h_1\,ds$')
    axes[1, 1].axhline(0, color=GREY, lw=0.7, alpha=0.65)
    axes[1, 1].scatter([L, L], [1, 0], color=[TEAL, PURPLE], s=26, zorder=4)
    axes[1, 1].set(title='(d) Accumulated gain errors',
                   xlabel=r'$z$', ylabel='integrated mode')
    axes[1, 1].legend(frameon=False, loc='lower left')
    for ax in axes.flat:
        clean(ax)
    save(fig, 'fig1_model')

    # Fig. 2: fixed terminal energy, different terminal waveform.
    cache = ROOT / 'results/illustrative_gain_fields.npz'
    xi = np.zeros((3, 4))
    xi[:, 3] = [-1, 0, 1]
    t, dt, k, fields = propagate(xi, .2)
    np.savez_compressed(cache, t=t, fields=fields, xi=xi, epsilon=.2)
    s = exact(t, L, 0.)
    silent = res['energy_silent']
    xx = .2 * np.array(silent['xi'])
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 4.7), layout='constrained')
    mask = np.abs(t - .9) <= 11
    for q, col in zip([-.2, 0, .2], [BLUE, GREY, ORANGE]):
        i = int(round(q / .2)) + 1
        axes[0, 0].plot(t[mask], np.abs(fields[i, mask]) ** 2, color=col,
                        ls='--' if q == 0 else '-', label=rf'$\epsilon\xi_4={q:g}$')
        aligned = fields[i] * np.exp(-1j * np.angle(np.vdot(s, fields[i])))
        phase = np.unwrap(np.angle(aligned[mask] * np.conj(s[mask])))
        axes[0, 1].plot(t[mask], phase, color=col, ls='--' if q == 0 else '-')
    axes[0, 0].set(title='(a) Terminal pulse intensity',
                   xlabel=r'$t$', ylabel=r'$|u(L,t)|^2$')
    axes[0, 0].legend(frameon=False)
    axes[0, 1].set(title='(b) Residual temporal phase',
                   xlabel=r'$t$', ylabel=r'phase difference (rad)')
    axes[0, 1].axhline(0, color=GREY, lw=0.7, alpha=0.65)
    width0 = res['responses']['0.0']['width']
    for values, den, col, lab, ls in [
        (silent['energy'], .5, TEAL, r'$E/E_0$', '--'),
        (silent['width'], width0, BLUE, r'$\sigma_t/\sigma_{t,0}$', '-'),
        (silent['peak'], max(abs(s) ** 2), ORANGE, r'$I_{\max}/I_{\max,0}$', '-'),
    ]:
        axes[1, 0].plot(xx, np.array(values) / den, color=col, label=lab, ls=ls)
    axes[1, 0].set(title='(c) Equal energy, different observables',
                   xlabel=r'$\epsilon\xi_4$', ylabel='normalized output')
    axes[1, 0].legend(frameon=False, loc='upper left')
    axes[1, 1].plot(xx, 1e4 * np.array(silent['aligned_loss']),
                    color=PURPLE, label='direct propagation')
    h44 = res['responses']['0.0']['aligned_gram'][3][3]
    axes[1, 1].plot(xx, 1e4 * xx ** 2 * h44, '--', color=GREY,
                    label='quadratic tangent')
    axes[1, 1].set(title='(d) Waveform loss at fixed energy',
                   xlabel=r'$\epsilon\xi_4$',
                   ylabel=r'corrected infidelity ($\times10^{-4}$)')
    axes[1, 1].legend(frameon=False)
    for ax in axes.flat:
        clean(ax)
    save(fig, 'fig2_energy_blind')

    # Fig. 3: range of the quadratic model and response alignment.
    eps = np.array(res['parameters']['epsilons'])

    def collect(a, e):
        return np.array([r['mean']['aligned_loss'] for r in res['records']
                         if r['control'] == a and r['epsilon'] == e])

    fig = plt.figure(figsize=(8.8, 3.15))
    fig.subplots_adjust(left=0.07, right=0.86, bottom=0.18, top=0.87, wspace=0.30)
    ax1 = fig.add_subplot(1, 2, 1)
    for a, col, marker, label in [(0., BLUE, 'o', 'baseline'),
                                  (best, ORANGE, 's', 'selected schedule')]:
        vals = np.array([collect(a, e).mean() for e in eps])
        err = np.array([collect(a, e).std(ddof=1) /
                        np.sqrt(res['parameters']['replicates']) for e in eps])
        pred = res['responses'][str(a)]['aligned_trace'] / 3
        ax1.errorbar(eps, vals / eps ** 2, yerr=2 * err / eps ** 2,
                     fmt=marker + '-', ms=4.4, color=col, label=label,
                     capsize=2.2, elinewidth=1.0)
        ax1.axhline(pred, color=col, ls='--', lw=1.2)
    tangent = Line2D([0], [0], color=GREY, ls='--', lw=1.2, label='tangent limit')
    ax1.set(xlabel=r'tolerance $\epsilon$',
            ylabel=r'$\mathbb{E}[1-F]/\epsilon^2$',
            title='(a) Departure from quadratic risk')
    ax1.legend(frameon=False, loc='lower left',
               handles=ax1.get_legend_handles_labels()[0] + [tangent],
               labels=ax1.get_legend_handles_labels()[1] + ['tangent limit'])
    clean(ax1)

    ax2 = fig.add_subplot(1, 2, 2)
    H = np.array(res['responses']['0.0']['aligned_gram'])
    norm = np.sqrt(np.diag(H))
    corr = H / (norm[:, None] * norm[None, :])
    im = ax2.imshow(corr, vmin=-1, vmax=1, cmap=CMAP_DIV)
    for i in range(4):
        for j in range(4):
            ax2.text(j, i, f'{corr[i, j]:.2f}', ha='center', va='center',
                     color='white' if abs(corr[i, j]) > .65 else '0.12',
                     fontsize=9)
    labs = [r'$\beta$', r'$\gamma$', r'$h_0$', r'$h_1$']
    ax2.set(xticks=range(4), yticks=range(4), xticklabels=labs,
            yticklabels=labs, title='(b) Projected response alignment')
    ax2.tick_params(length=0)
    cax = fig.add_axes([0.885, 0.22, 0.014, 0.59])
    colorbar(fig, im, cax, 'response correlation')
    save(fig, 'fig3_response')

    # Fig. 4: constrained design comparison.
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), layout='constrained')
    grid = res['design']['grid']
    js = np.array([x['aligned_trace'] / 3 for x in grid])
    aa = [x['control'] for x in grid]
    axes[0].plot(aa, js, color=BLUE)
    selected_j = res['design']['best']['aligned_trace'] / 3
    axes[0].scatter([best, 0], [selected_j, res['responses']['0.0']['aligned_trace'] / 3],
                    c=[ORANGE, BLUE], s=30, zorder=4)
    axes[0].annotate('selected grid point', xy=(best, selected_j),
                     xytext=(-.58, selected_j + .0065),
                     arrowprops={'arrowstyle': '-', 'color': GREY},
                     fontsize=8.4)
    axes[0].set(xlabel=r'schedule parameter $a$',
                ylabel=r'$J(a)=\mathrm{tr}(H^{\rm al})/3$',
                title='(a) Constrained schedule comparison')
    base, new = [], []
    for rep in range(res['parameters']['replicates']):
        for a, buf in [(0., base), (best, new)]:
            ar = np.load(ROOT / 'results' / f'samples_r{rep}_a{a:+.3f}_e0.100.npz')
            buf.extend(ar['aligned_loss'])
    gain = np.array(base) - np.array(new)
    x = np.sort(gain)
    y = np.arange(1, len(x) + 1) / len(x)
    axes[1].plot(x * 1e4, y, color=TEAL)
    axes[1].axvline(0, color=GREY, lw=0.9, ls='--')
    axes[1].set(xlabel=r'paired loss reduction ($\times10^{-4}$)',
                ylabel='empirical cumulative probability',
                title=r'(b) Fresh samples at $\epsilon=0.1$')
    axes[1].text(.96, .13, f'{np.mean(gain > 0):.1%} of samples improve',
                 ha='right', transform=axes[1].transAxes, fontsize=8.4)
    for ax in axes:
        clean(ax)
    save(fig, 'fig4_design')
    print('Created four 2D figures; paired improvement fraction', np.mean(gain > 0))


if __name__ == '__main__':
    build(json.loads((ROOT / 'results/study.json').read_text()))
