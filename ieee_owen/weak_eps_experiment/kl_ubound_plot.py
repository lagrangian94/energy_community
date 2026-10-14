"""
The out-of-sample bound against the training sample size, as a figure
(kl_experiment_plan.md, sec. 8.18 (6)): one panel per community size, |Omega| across,
EUR per member up on a log axis, three series:

  bound   what no coalition can gain per member by leaving, on the reference sample
          (U^ws of run_kl_ubound_ws.py; U of run_kl_ubound.py is in the CSV where a
          joint run exists)
  shift   what each member receives above the promise, (earned - promised) / n;
          its magnitude is drawn, hollow where the plan earned less than promised
  eps_K   the in-sample tolerance of the robust Owen solution

The bound is one to two orders above eps_K, so the figure shows magnitudes and their
trend, not a table of values. With several days a marker is the mean over the days
and the bar their range. A second row draws the bound alone on a linear axis, every
day as a line of its own.

Reads the main queue (|Omega| = 20), <out>/train (the other sample sizes,
run_kl_utrain.py) and <out>/ws_*.json; writes <out>/ubound_summary.csv and
<out>/ubound_trend.{pdf,png}. A (size, sample, day) with a file missing is left out.

    python ieee_owen/weak_eps_experiment/kl_ubound_plot.py --days 1
"""
import os, glob, json, argparse

import run_kl_main as Q                      # chdirs to the repo root

MAIN = Q.OUT
OUT = os.path.join('ieee_owen', 'weak_eps_experiment', 'stochastic', 'ubound17_xi')
PLAN_SCEN, REF = 20, 500
# the first three categorical slots of the palette (they separate under every pairing
# and for colour-blind readers); the markers repeat the identity for print
SERIES = (('bound', 'out-of-sample bound', '#2a78d6', 'o'),
          ('shift', 'shift per member (hollow: below the promise)', '#eb6834', 's'),
          ('eps_K', r'in-sample $\varepsilon_K$', '#1baf7a', '^'))


def one(pattern):
    found = glob.glob(pattern)
    return json.load(open(found[0], encoding='utf-8')) if len(found) == 1 else None


def point(out, n, scen, day):
    """Bound, shift and eps_K [EUR per member] of one (n, |Omega|, day); None if a
    file is missing."""
    tr = one(os.path.join(MAIN, f'{n}p_*_day{day}_S{scen}_*_main.json')) if scen == PLAN_SCEN \
        else one(os.path.join(out, 'train', f'{n}p_*_day{day}_S{scen}_seed0_*.json'))
    ws = one(os.path.join(out, f'ws_n{n}_S{PLAN_SCEN}_ref{REF}_day{day}.json'))
    if tr is None or ws is None or ws['done'] < REF:
        return None
    if scen >= PLAN_SCEN:                    # the main run's plan: what it earns is in ws
        earned = ws['earned']
    else:
        oos = one(os.path.join(out, 'train', f'oos_n{n}_S{scen}_day{day}.json'))
        if oos is None:
            return None
        earned = oos['plans']['dro']['oos_mean']
    star = {u: -v for u, v in tr['dw']['sigma'].items()}             # profit convention
    pay = {u: star[u] - (sum(star.values()) - earned) / n for u in star}
    r = {'n': n, 'scen': scen, 'day': day, 'radius': Q.radius(scen), 'earned': earned,
         'bound': max(ws['u_bar'][u] - pay[u] for u in star),
         'shift': (earned + tr['ef']['obj']) / n, 'eps_K': tr['allocation']['eps_LR'],
         'cg_status': tr['dw']['status'], 'U_joint': float('nan'), 'ref_joint': 0}
    for f in glob.glob(os.path.join(out, f'{n}p_*_day{day}_S*_seed1000_*_ref_S{PLAN_SCEN}.json')):
        ref = json.load(open(f, encoding='utf-8'))
        if scen >= PLAN_SCEN and ref['scenarios'] > r['ref_joint']:  # the largest joint run
            e = -ref['ef']['obj']
            r['U_joint'] = max(-ref['dw']['sigma_bound'][u] - star[u]
                               + (sum(star.values()) - e) / n for u in star)
            r['ref_joint'] = ref['scenarios']
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', default='6,15,30,60')
    ap.add_argument('--scen', default='5,20,40,100')
    ap.add_argument('--days', default='1')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    sizes = [int(v) for v in a.n.split(',')]
    scens = [int(v) for v in a.scen.split(',')]
    pts = [p for n in sizes for m in scens for d in Q.days(a.days)
           if (p := point(a.out, n, m, d)) is not None]
    cols = ('n', 'scen', 'day', 'radius', 'earned', 'bound', 'U_joint', 'ref_joint', 'shift',
            'eps_K', 'cg_status')
    with open(os.path.join(a.out, 'ubound_summary.csv'), 'w') as f:
        f.write(','.join(cols) + '\n')
        for p in pts:
            f.write(','.join(f'{p[c]:.4f}' if isinstance(p[c], float) else str(p[c])
                             for c in cols) + '\n')
    print(open(os.path.join(a.out, 'ubound_summary.csv')).read())

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 8, 'axes.linewidth': 0.6, 'pdf.fonttype': 42})
    fig, grid = plt.subplots(2, len(sizes), figsize=(7.16, 4.0), sharey='row', sharex=True)
    axes, low = grid
    for ax, n in zip(axes, sizes):
        for key, label, color, marker in SERIES:
            xs, mid, lo, hi, neg = [], [], [], [], []
            for m in scens:
                v = [p[key] for p in pts if p['n'] == n and p['scen'] == m]
                if v:
                    mean = sum(v) / len(v)
                    xs.append(m); mid.append(abs(mean)); neg.append(mean < 0)
                    lo.append(min(abs(x) for x in v)); hi.append(max(abs(x) for x in v))
            if not xs:
                continue
            # the log axis takes magnitudes; only the shift can be negative (the plan
            # earned less than promised), and such a point is drawn hollow
            ax.plot(xs, mid, color=color, marker=marker, lw=1.5, ms=4.5, label=label,
                    markeredgecolor='white', markeredgewidth=0.6)
            ax.plot([x for x, g in zip(xs, neg) if g], [y for y, g in zip(mid, neg) if g],
                    ls='none', marker=marker, ms=4.5, markerfacecolor='white',
                    markeredgecolor=color, markeredgewidth=1.0)
            if any(h > l for l, h in zip(lo, hi)):
                ax.vlines(xs, lo, hi, color=color, lw=1.0)
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xticks(scens); ax.set_xticklabels([str(m) for m in scens]); ax.minorticks_off()
        ax.set_xlim(min(scens) / 1.3, max(scens) * 1.3)
        ax.set_title(f'$n={n}$', fontsize=8)
        ax.grid(True, which='major', lw=0.3, color='#d9d8d2')
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('EUR per member')
    # the bound alone on a linear axis: the days differ in level by more than the
    # bound moves with |Omega|, so every day is drawn (thin) under the mean of the
    # days that have all the sample sizes
    color = SERIES[0][2]
    for ax, n in zip(low, sizes):
        by_day = {}
        for p in pts:
            if p['n'] == n:
                by_day.setdefault(p['day'], {})[p['scen']] = p['bound']
        for d, v in sorted(by_day.items()):
            xs = [m for m in scens if m in v]
            ax.plot(xs, [v[m] for m in xs], color='#9bbde8', lw=0.8, marker='o', ms=2.2,
                    label='each day' if d == min(by_day) else None)
        full = [v for v in by_day.values() if all(m in v for m in scens)]
        if full:
            ax.plot(scens, [sum(v[m] for v in full) / len(full) for m in scens], color=color,
                    lw=1.8, marker='o', ms=4.5, markeredgecolor='white', markeredgewidth=0.6,
                    label=f'mean of {len(full)} days')
        ax.set_xscale('log')
        ax.set_xticks(scens); ax.set_xticklabels([str(m) for m in scens]); ax.minorticks_off()
        ax.set_xlim(min(scens) / 1.3, max(scens) * 1.3)
        ax.set_xlabel(r'training scenarios $|\Omega|$')
        ax.grid(True, which='major', lw=0.3, color='#d9d8d2')
        ax.legend(frameon=False, fontsize=6.5, loc='upper right', handlelength=1.5)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    low[0].set_ylim(0, 1.05 * max(p['bound'] for p in pts))      # shared along the row
    low[0].set_ylabel('bound, EUR per member')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(SERIES), frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(a.out, f'ubound_trend.{ext}'), dpi=300)
    print('wrote', os.path.join(a.out, 'ubound_trend.{pdf,png}'))


if __name__ == '__main__':
    main()
