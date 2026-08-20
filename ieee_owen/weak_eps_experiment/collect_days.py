"""Collect the per-day CG results into `eps_31day.json`, the source for Table `tab:results`.

Reads `maximin_{n}p_day{d}.json` -- from `maximin_slack1e-7/` for the days produced before
the (R2) fix, otherwise from this directory -- and keeps only the fields upstream of the
maximin selection, which the (R2) defect did not touch: the grand-coalition MILP value, the
Dantzig-Wolfe master value, their gap, and the two solve times.

January has 31 days and `run_multiday.DAYS` is `range(1, 32)`, so 31 is the day count the
rest of the harness uses; anything reported over 30 is a truncation of it.
"""
import os, json, argparse

OUT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(OUT, 'maximin_slack1e-7')


def find(n, day):
    for d in (SRC, OUT):
        p = os.path.join(d, f'maximin_{n}p_day{day}.json')
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=31)
    ap.add_argument('--out', default='eps_31day.json')
    a = ap.parse_args()

    out = {}
    for n in (6, 15, 30):
        rows = []
        for d in range(1, a.days + 1):
            p = find(n, d)
            if p is None:
                print(f"  !! n={n} day {d}: missing")
                continue
            j = json.load(open(p))
            rows.append(dict(day=d, omega=abs(j['gap']), eps=j['eps_bound_gap_over_N'],
                             v_mip=j['v_mip'], v_chp=j['v_chp'],
                             t_mip=j['time_mip_s'], t_cg=j['time_cg_s']))
        out[str(n)] = rows
        print(f"n={n:>2}: {len(rows)} days")
    p = os.path.join(OUT, a.out)
    json.dump(out, open(p, 'w'), indent=1)
    print(f"-> {p}")


if __name__ == '__main__':
    main()
