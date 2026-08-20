"""
Coarser settlement resolution: collapse a 24-hour parameter set to |T| periods.

WHY. The coupling row count is m = (|K|+3)|T|, and Proposition `prop:eps` only starts
to bite once n > m+1. At |T| = 24 that boundary sits at n = 145, far beyond any
community we can solve, so the decay regime is never entered and the proposition
cannot be tested on its own terms. Varying |T| moves the boundary:

    |T| = 24 -> m = 144, boundary n > 145   (unreachable)
    |T| =  8 -> m =  48, boundary n >  49
    |T| =  4 -> m =  24, boundary n >  25   (n = 30 is inside)
    |T| =  2 -> m =  12, boundary n >  13   (n = 15, 30 inside)

NOT a data hack. |T| is the number of settlement periods, which the draft itself calls
a property of the market design ("one row per carrier and period"), not of the data. A
community settling four times a day rather than twenty-four is a different market
design, and that is exactly the axis the proposition is about.

WHY NOT TRUNCATE. Taking a 6-hour window out of the day instead would break three
things: storage loses the daily arbitrage cycle it exists for, the electrolyzer's
commitment pattern (start once, run through the cheap hours) collapses to a single
on/off decision, and the answer depends entirely on which window is picked. Averaging
into blocks keeps the whole day -- day/night structure, storage cycle, commitment
pattern -- and only coarsens time.

Every per-period quantity here is a rate (MW, EUR/MWh, utility per MWh), so the block
statistic is the MEAN, not the sum. Each aggregated period is then one settlement
interval, and the model is used unchanged.
"""
import re
import numpy as np

_PER_T_KEY = re.compile(r'^(.*)_(\d+)$')
_PRICE_DICTS = ('pi_E_gri', 'pi_G_gri', 'pi_H_gri')


def blocks_for(n_periods, base_hours=24):
    if base_hours % n_periods:
        raise ValueError(f"{base_hours} hours does not divide into {n_periods} periods")
    dt = base_hours // n_periods
    return [list(range(k * dt, (k + 1) * dt)) for k in range(n_periods)]


def aggregate_params(params, n_periods, base_hours=24, verbose=False):
    """Return a copy of `params` re-expressed on `n_periods` settlement intervals.

    Only complete 0..base_hours-1 series are touched; any key whose numeric suffix does
    not form a full series is left alone, so unrelated keys that happen to end in a
    digit cannot be mangled.
    """
    if n_periods == base_hours:
        return dict(params)
    blocks = blocks_for(n_periods, base_hours)
    out = dict(params)

    # --- scalar per-period keys:  <prefix>_<t> ---
    groups = {}
    for k, v in params.items():
        m = _PER_T_KEY.match(k)
        if not m:
            continue
        t = int(m.group(2))
        if not (0 <= t < base_hours):
            continue
        if not isinstance(v, (int, float, np.integer, np.floating)):
            continue
        groups.setdefault(m.group(1), {})[t] = float(v)

    done = []
    for prefix, series in groups.items():
        if set(series) != set(range(base_hours)):
            continue                      # partial series -> not a time index
        for t in range(base_hours):
            out.pop(f'{prefix}_{t}', None)
        for bi, b in enumerate(blocks):
            out[f'{prefix}_{bi}'] = float(np.mean([series[t] for t in b]))
        done.append(prefix)

    # --- array-valued market price dicts ---
    for key in _PRICE_DICTS:
        d = params.get(key)
        if isinstance(d, dict):
            out[key] = {kk: [float(np.mean([vv[t] for t in b])) for b in blocks]
                        for kk, vv in d.items()
                        if hasattr(vv, '__len__') and len(vv) >= base_hours}
            done.append(key)

    if verbose:
        print(f"  [horizon] {base_hours}h -> {n_periods} periods "
              f"({base_hours // n_periods}h blocks), {len(done)} series aggregated")
    return out


def expected_m(n_periods, enable_reserve=True, enable_peak=True):
    """m = (|K|+3)|T| in the draft's notation: 3 carrier balances + 2 reserve + 1 peak."""
    rows = 3 + (2 if enable_reserve else 0) + (1 if enable_peak else 0)
    return rows * n_periods


def regime_boundary(n_periods, **kw):
    """prop:eps decay needs n > m+1."""
    return expected_m(n_periods, **kw) + 1
