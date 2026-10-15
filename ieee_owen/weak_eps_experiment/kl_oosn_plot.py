"""
The out-of-sample budget against the training sample size, as a table and a figure
(kl_experiment_plan.md, sec. 8.20): for every community size, the shift each member
receives, (earned - promised) / n, with the ball and without it (r = 0), at
|Omega| = 5, 20, 40, 100. Both settle closer to what was promised as the sample grows,
and closer to each other, the radius being of order 1 / |Omega|.

Reads run_kl_oos.py's results (|Omega| = 20) and run_kl_oosn.py's (the others); with
--improve, a cell of |Omega| > 20 is read from run_kl_oosn_improve.py's results instead
(the plan re-chosen on its own sample) where that file exists. Writes <out>/oosn_days.csv
(one row per size, sample, day and plan), <out>/oosn_summary.csv and the figure: one
column wide, 2 x 2 (<out>/oosn_shift.{pdf,png}), and two columns wide, 1 x 4
(<out>/oosn_shift_row.{pdf,png}). A marker is the mean over the days, the band the
middle half of the days.

    python ieee_owen/weak_eps_experiment/kl_oosn_plot.py [--improve]
"""
import os, json, argparse

import numpy as np

import run_kl_main as Q                      # chdirs to the repo root
import run_kl_oos as O

OUT = os.path.join('ieee_owen', 'weak_eps_experiment', 'stochastic', 'oosn17_xi')
PLAN_SCEN = 20
# the first two categorical slots of the palette; the markers repeat the identity
# legend names follow the manuscript's DP^DR and DP^SAA
SERIES = (('dro', r'DR, $r=\chi^2_{1,0.95}/(2|\Omega|)$', '#2a78d6', 'o'),
          ('saa', r'SAA, $r=0$', '#eb6834', 's'))
# (file stem, panel grid, figure size [in]): IEEE one column is 3.5 in, two are 7.16 in
LAYOUTS = (('oosn_shift', (2, 2), (3.5, 3.4)), ('oosn_shift_row', (1, 4), (7.16, 2.1)))


def draw(stat, sizes, scens, rows_cols, figsize, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(*rows_cols, figsize=figsize, sharex=True, sharey=True,
                             squeeze=False)
    for ax, n in zip(axes.flat, sizes):
        ax.axhline(0.0, color='#52514e', lw=0.6)
        for lab, label, color, marker in SERIES:
            xs = [m for m in scens if (n, m, lab) in stat]
            if not xs:
                continue
            mean, lo, hi = (np.array([stat[n, m, lab][i] for m in xs]) for i in range(3))
            ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)
            ax.plot(xs, mean, color=color, marker=marker, lw=1.3, ms=4, label=label,
                    markeredgecolor='white', markeredgewidth=0.5)
        ax.set_xscale('log')
        ax.set_xticks(scens); ax.set_xticklabels([str(m) for m in scens]); ax.minorticks_off()
        ax.set_xlim(min(scens) / 1.3, max(scens) * 1.3)
        ax.text(0.96, 0.95, f'$n={n}$', transform=ax.transAxes, ha='right', va='top')
        ax.grid(True, which='major', lw=0.3, color='#d9d8d2')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    # IEEE style: sentence case, units in parentheses. The caption gives the formula,
    # (val(DP^out_N) - val(DP^DR_N)) / n at each series' radius (r = 0 is DP^SAA), in the
    # manuscript's own macros rather than in matplotlib's fonts
    fig.supxlabel(r'Sample size $|\Omega|$', fontsize=8)
    fig.supylabel('Shift per member (€)', fontsize=8, x=0.01)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(SERIES), frameon=False,
               fontsize=7, handlelength=1.6, columnspacing=1.0, bbox_to_anchor=(0.5, 1.0))
    # room for the shared labels: 0.17 in on the left, 0.28 in on top for the legend
    fig.tight_layout(rect=(0.17 / figsize[0], 0, 1, 1 - 0.28 / figsize[1]), pad=0.3,
                     h_pad=0.4, w_pad=0.4)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{path}.{ext}', dpi=300)
    plt.close(fig)
    print('wrote', f'{path}.{{pdf,png}}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', default='6,15,30,60')
    ap.add_argument('--scen', default='5,20,40,100')
    ap.add_argument('--days', default='1-31')
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--improve', action='store_true',
                    help='|Omega| > 20 from the re-chosen plans (<out>/improve) where present')
    a = ap.parse_args()
    sizes = [int(v) for v in a.n.split(',')]
    scens = [int(v) for v in a.scen.split(',')]
    rows, improved = [], 0
    for n in sizes:
        for m in scens:
            for d in Q.days(a.days):
                f = os.path.join(O.OUT if m == PLAN_SCEN else a.out, f'oos_n{n}_S{m}_day{d}.json')
                g = os.path.join(a.out, 'improve', f'oos_n{n}_S{m}_day{d}.json')
                if a.improve and m > PLAN_SCEN and os.path.exists(g):
                    f, improved = g, improved + 1
                if not os.path.exists(f):
                    continue
                plans = json.load(open(f, encoding='utf-8'))['plans']
                for lab in ('dro', 'saa'):
                    if lab in plans:
                        p = plans[lab]
                        rows.append({'n': n, 'scen': m, 'day': d, 'plan': lab,
                                     'promised': p['promised'], 'in_mean': p['in_mean'],
                                     'earned': p['oos_mean'],
                                     'shift': (p['oos_mean'] - p['promised']) / n,
                                     'hedge': (p['in_mean'] - p['promised']) / n})
    with open(os.path.join(a.out, 'oosn_days.csv'), 'w') as f:
        cols = ('n', 'scen', 'day', 'plan', 'promised', 'in_mean', 'earned', 'shift', 'hedge')
        f.write(','.join(cols) + '\n')
        for r in rows:
            f.write(','.join(f'{r[c]:.4f}' if isinstance(r[c], float) else str(r[c])
                             for c in cols) + '\n')
    stat = {}
    lines = ['n,scen,plan,days,budget_holds,shift_mean,shift_abs_mean,shift_sd,shift_q25,'
             'shift_q75,hedge_mean']
    for n in sizes:
        for m in scens:
            for lab in ('dro', 'saa'):
                v = np.array([r['shift'] for r in rows
                              if (r['n'], r['scen'], r['plan']) == (n, m, lab)])
                if not len(v):
                    continue
                h = np.mean([r['hedge'] for r in rows
                             if (r['n'], r['scen'], r['plan']) == (n, m, lab)])
                stat[n, m, lab] = (v.mean(), *np.percentile(v, [25, 75]))
                lines.append(f'{n},{m},{lab},{len(v)},{int((v >= 0).sum())},{v.mean():.3f},'
                             f'{np.abs(v).mean():.3f},{v.std(ddof=1) if len(v) > 1 else 0:.3f},'
                             f'{stat[n, m, lab][1]:.3f},{stat[n, m, lab][2]:.3f},{h:.3f}')
    with open(os.path.join(a.out, 'oosn_summary.csv'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    if a.improve:
        print(f'{improved} cells of |Omega| > {PLAN_SCEN} from the re-chosen plans')
    for stem, grid, size in LAYOUTS:
        draw(stat, sizes, scens, grid, size, os.path.join(a.out, stem))


if __name__ == '__main__':
    main()
