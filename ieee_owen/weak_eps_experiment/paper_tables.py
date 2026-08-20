"""Regenerate every number in `ieee_draft.txt` Sec. `sec:num` from the stored results.

The point is auditability: each figure in the two tables should be traceable to a file on
disk rather than to a transcript. Run this and diff against the draft.

    python weak_eps_experiment/paper_tables.py            # print both tables
    python weak_eps_experiment/paper_tables.py --csv      # also write paper_tables.csv

Sources, and what each is good for:

  eps_31day.json     v^MIP, v^CHP, omega, eps^LR and the MIP/CG timings for 30 days at each
                     size. Extracted from the per-day runs under `maximin_slack1e-7/`; the
                     (R2) defect recorded in `validation_plan.md` affected only the maximin
                     selection in those files, not these fields.
  excess_<n>p.json   per-capita excess of the executed allocation per day (measure_excess.py)
  cg_<n>p.json       single-instance run of run_experiment.py -- used only for the
                     row-generation column of the runtime table
  baseline_<n>p/rowgen.csv
                     row generation over the day sweep, one row per day. PREFERRED when
                     present. A `rowgen.csv` in a run folder is by convention current
                     (4-hour reserve product); the pre-4-hour files are parked next to it
                     as `rowgen_stale_24h_product.csv` so they cannot be picked up.
  rowgen_<n>p.json   fallback: row generation on ONE instance, for the sizes that have no
                     sweep. Convergence, time, coalition cuts.

Row generation reports a geometric mean over the instances that RETURNED A CERTIFICATE,
alongside how many did. Averaging a censored run in with the rest would report the budget
rather than the algorithm: a day cut off at 3600 s did not take 3600 s, it took longer
than that by an unknown margin.

Aggregates are geometric means, which is what the draft reports: the object of interest is
how a quantity scales in `n`, and a geometric mean is the summary that respects that.
"""
import os, sys, json, math, argparse, statistics as st, csv as _csv

OUT = os.path.dirname(os.path.abspath(__file__))
SIZES = (6, 15, 30)


def gmean(v):
    return math.exp(sum(math.log(x) for x in v) / len(v))


def load_rowgen(n):
    """Row generation at size `n`: the day sweep if there is one, else one instance.

    Returns the same keys either way, so the table does not care which it got:
      days, certified          how many instances, how many returned a certificate
      time_certified, cuts     geometric means over the CERTIFIED instances only
      cuts_cutoff              geometric mean cuts over the cut-off instances (None if
                               there are none) -- a text remark, not a table row
      budget                   the per-day budget the cut-off instances hit
    """
    sweep = os.path.join(OUT, f'baseline_{n}p', 'rowgen.csv')
    if os.path.exists(sweep):
        with open(sweep) as fh:
            rows = list(_csv.DictReader(fh))
        ok = [r for r in rows if r['converged'] == 'True']
        no = [r for r in rows if r['converged'] != 'True']
        return dict(
            days=len(rows), certified=len(ok),
            time_certified=gmean([float(r['time_rowgen_s']) for r in ok]) if ok else None,
            cuts=gmean([int(r['n_coalitions']) for r in ok]) if ok else None,
            cuts_cutoff=gmean([int(r['n_coalitions']) for r in no]) if no else None,
            budget=max((float(r['time_rowgen_s']) for r in no), default=None))
    j = json.load(open(os.path.join(OUT, f'rowgen_{n}p.json')))
    conv = bool(j['converged'])
    return dict(days=1, certified=int(conv),
                time_certified=j['time_rowgen_s'] if conv else None,
                cuts=j['n_coalitions_generated'] if conv else None,
                cuts_cutoff=None if conv else j['n_coalitions_generated'],
                budget=None if conv else j['time_rowgen_s'])


def load():
    day = json.load(open(os.path.join(OUT, 'eps_31day.json')))
    exc = {n: json.load(open(os.path.join(OUT, f'excess_{n}p.json'))) for n in SIZES}
    cg = {n: json.load(open(os.path.join(OUT, f'cg_{n}p.json'))) for n in SIZES}
    rg = {n: load_rowgen(n) for n in SIZES}
    return day, exc, cg, rg


def build():
    day, exc, cg, rg = load()
    rows = {}
    for n in SIZES:
        d = day[str(n)]
        e = [exc[n][k] for k in sorted(exc[n], key=int) if 'excess' in exc[n][k]]
        outside = [r for r in e if len(r['coalition']) > 0]
        share = [max(0.0, r['excess']) / r['eps'] for r in e]
        rows[n] = dict(
            days=len(d),
            v_mip=gmean([abs(x['v_mip']) for x in d]),
            omega=gmean([x['omega'] for x in d]),
            eps=gmean([x['eps'] for x in d]),
            omega_min=min(x['omega'] for x in d), omega_max=max(x['omega'] for x in d),
            excess_days=len(e), in_core=len(e) - len(outside),
            excess=gmean([r['excess'] for r in outside]) if outside else None,
            share_median=st.median(share), share_max=max(share),
            t_mip=gmean([x['t_mip'] for x in d]), t_cg=gmean([x['t_cg'] for x in d]),
            rg_days=rg[n]['days'], rg_certified=rg[n]['certified'],
            rg_time=rg[n]['time_certified'], rg_cuts=rg[n]['cuts'],
            rg_cuts_cutoff=rg[n]['cuts_cutoff'], rg_budget=rg[n]['budget'],
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', action='store_true')
    a = ap.parse_args()
    r = build()
    hdr = "".join(f"{'n=' + str(n):>12}" for n in SIZES)

    def line(label, fn):
        print(f"{label:<38}" + "".join(f"{fn(r[n]):>12}" for n in SIZES))

    print(f"\ntab:results   (geometric means over {r[6]['days']} daily instances per size)")
    print(f"{'':<38}" + hdr)
    line('v^MIP(N)',                lambda x: f"{x['v_mip']:.1f}")
    line('omega^LR',                lambda x: f"{x['omega']:.2f}")
    line('eps^LR = omega/n',        lambda x: f"{x['eps']:.2f}")
    line('days in the exact core',  lambda x: f"{x['in_core']}/{x['excess_days']}")
    line('eps(chi) on the rest',    lambda x: '---' if x['excess'] is None else f"{x['excess']:.2f}")
    print("  -- reported in the text or caption, not as table rows --")
    line('  share of the guarantee spent, median',
                                    lambda x: f"{x['share_median'] * 100:.0f}%")
    line('  share of the guarantee spent, max',
                                    lambda x: f"{x['share_max'] * 100:.0f}%")
    line('  omega range over the days',
                                    lambda x: f"[{x['omega_min']:.2f},{x['omega_max']:.1f}]")

    print(f"\ntab:runtime   (geometric means; row generation over CERTIFIED instances only)")
    print(f"{'':<38}" + hdr)
    line('grand-coalition MILP [s]', lambda x: f"{x['t_mip']:.1f}")
    line('column generation [s]',    lambda x: f"{x['t_cg']:.1f}")
    line('row gen: certified',       lambda x: f"{x['rg_certified']}/{x['rg_days']}")
    line('row gen: time, certified [s]',
         lambda x: '---' if x['rg_time'] is None else
                   (f"{x['rg_time']:.1f}" if x['rg_time'] < 100 else f"{x['rg_time']:.0f}"))
    line('  coalition cuts, certified',
         lambda x: '---' if x['rg_cuts'] is None else f"{x['rg_cuts']:.0f}")
    print("  -- reported in the text or caption, not as table rows --")
    line('  coalition cuts, cut off',
         lambda x: '---' if x['rg_cuts_cutoff'] is None else f"{x['rg_cuts_cutoff']:.0f}")
    line('  budget the cut-off runs hit [s]',
         lambda x: '---' if x['rg_budget'] is None else f"{x['rg_budget']:.0f}")

    if a.csv:
        import csv
        path = os.path.join(OUT, 'paper_tables.csv')
        with open(path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['quantity'] + [f'n={n}' for n in SIZES])
            for k in sorted(r[6]):
                w.writerow([k] + [r[n][k] for n in SIZES])
        print(f"\n-> {path}")


if __name__ == '__main__':
    main()
