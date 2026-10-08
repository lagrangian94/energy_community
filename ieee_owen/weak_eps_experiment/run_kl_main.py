"""
Main KL-DRO runs of the uncertainty manuscript (ieee_owen/kl_experiment_plan.md, sec. 8):
stochastic_extension.py for every (|Omega|, day, n), one at a time, at the
Duchi-Glynn-Namkoong radius r = chi2_{1,0.95} / (2 |Omega|).

Each run gets its own log, EF cache and EF Gurobi log under --out; a run whose log
already says 'wrote' is skipped, so the queue can be stopped and restarted. The
solver's memory is polled every 30 s (the venv python.exe is a launcher, so the
child interpreter is the one measured); if the machine's available memory falls
below --min-free GB the run is killed and the queue moves on. One row per run goes
to <out>/runs.csv.

    python ieee_owen/weak_eps_experiment/run_kl_main.py                 # whole queue
    python ieee_owen/weak_eps_experiment/run_kl_main.py --scen 20 --n 6,15 --days 1-3
"""
import os, sys, time, argparse, subprocess, ctypes, ctypes.wintypes as wt

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
os.chdir(_ROOT)

from scipy.stats import chi2

OUT = os.path.join('ieee_owen', 'weak_eps_experiment', 'stochastic', 'main17_xi')
PRICE_SCALE = 17.0 / 56.0
ALPHA = 0.05
POLL = 30

k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
k32.OpenProcess.restype = wt.HANDLE
k32.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
k32.CreateToolhelp32Snapshot.restype = wt.HANDLE
psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD]


class PMC(ctypes.Structure):
    _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('a', ctypes.c_size_t), ('b', ctypes.c_size_t), ('c', ctypes.c_size_t),
                ('d', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t),
                ('PeakPagefileUsage', ctypes.c_size_t)]


class MSX(ctypes.Structure):
    _fields_ = [('dwLength', wt.DWORD), ('dwMemoryLoad', wt.DWORD)] + \
               [(f'u{i}', ctypes.c_ulonglong) for i in range(7)]


class PE32(ctypes.Structure):
    _fields_ = [('dwSize', wt.DWORD), ('cntUsage', wt.DWORD), ('th32ProcessID', wt.DWORD),
                ('th32DefaultHeapID', ctypes.c_size_t), ('th32ModuleID', wt.DWORD),
                ('cntThreads', wt.DWORD), ('th32ParentProcessID', wt.DWORD),
                ('pcPriClassBase', ctypes.c_long), ('dwFlags', wt.DWORD),
                ('szExeFile', ctypes.c_char * 260)]


def avail_gb():
    m = MSX(); m.dwLength = ctypes.sizeof(m)
    k32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.u1 / 2**30                                  # ullAvailPhys


def child_of(pid):
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)
    e = PE32(); e.dwSize = ctypes.sizeof(e)
    ok = k32.Process32First(snap, ctypes.byref(e))
    found = None
    while ok:
        if e.th32ParentProcessID == pid:
            found = e.th32ProcessID
            break
        ok = k32.Process32Next(snap, ctypes.byref(e))
    k32.CloseHandle(snap)
    return found


def mem(handle):
    c = PMC(); c.cb = ctypes.sizeof(c)
    psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb)
    return c


def radius(scen):
    return float(chi2.ppf(1 - ALPHA, 1) / (2 * scen))


def run_one(n, scen, day, a):
    tag = f'n{n}_S{scen}_day{day}'
    log = os.path.join(a.out, 'logs', f'{tag}.log')
    if os.path.exists(log) and 'wrote ' in open(log, encoding='utf-8', errors='replace').read():
        return None
    os.makedirs(os.path.dirname(log), exist_ok=True)
    cmd = [sys.executable, os.path.join('ieee_owen', 'stochastic_extension.py'),
           '--n', str(n), '--scenarios', str(scen), '--day', str(day),
           '--kl-radius', repr(radius(scen)), '--kl-master', 'dual',
           '--reserve-price-scale', repr(PRICE_SCALE), '--stall-barrier',
           '--mip-time-limit', str(a.ef_round_limit), '--ef-nodefile-start', str(a.nodefile),
           '--ef-cache', os.path.join(a.out, 'ef_cache', f'{tag}.pkl'),
           '--ef-log', os.path.join(a.out, 'logs', f'{tag}_ef_gurobi.log'),
           '--tag', 'main', '--out', a.out]
    env = dict(os.environ, PYTHONHASHSEED='21', PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        kid, h = None, None
        peak_priv = peak_ws = 0.0
        min_free, killed = 99.0, False
        while p.poll() is None:
            if h is None:                    # the interpreter may not be spawned yet
                kid = child_of(p.pid)
                if kid:
                    h = k32.OpenProcess(0x1410, False, kid)
            if h:
                c = mem(h)
                peak_priv = max(peak_priv, c.PeakPagefileUsage / 2**30)
                peak_ws = max(peak_ws, c.PeakWorkingSetSize / 2**30)
            free = avail_gb()
            min_free = min(min_free, free)
            if free < a.min_free:
                p.kill()
                if kid:
                    subprocess.run(['taskkill', '/F', '/T', '/PID', str(kid)],
                                   capture_output=True)
                killed = True
                break
            time.sleep(POLL if h else 2)
        rc = p.wait()
    if h:
        c = mem(h)
        peak_priv = max(peak_priv, c.PeakPagefileUsage / 2**30)
        peak_ws = max(peak_ws, c.PeakWorkingSetSize / 2**30)
    row = (f'{time.strftime("%Y-%m-%d %H:%M")},{n},{scen},{day},{radius(scen):.6g},{rc},'
           f'{time.time() - t0:.0f},{peak_priv:.2f},{peak_ws:.2f},{min_free:.2f},{int(killed)}')
    csv = os.path.join(a.out, 'runs.csv')
    new = not os.path.exists(csv)
    with open(csv, 'a') as f:
        if new:
            f.write('finished,n,scen,day,r,exit,wall_s,peak_private_gb,peak_ws_gb,'
                    'min_avail_gb,killed\n')
        f.write(row + '\n')
    return row


def days(spec):
    out = []
    for part in spec.split(','):
        lo, _, hi = part.partition('-')
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scen', default='20,10,5', help='outer loop, in this order')
    ap.add_argument('--days', default='1-31', help='middle loop, e.g. 1-31 or 3,6,9')
    ap.add_argument('--n', default='6,15,30,60', help='inner loop')
    ap.add_argument('--ef-round-limit', type=float, default=10800)
    ap.add_argument('--nodefile', type=float, default=3.0, help='EF NodefileStart [GB]')
    ap.add_argument('--min-free', type=float, default=0.7, help='kill below this [GB]')
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()
    for scen in [int(s) for s in a.scen.split(',')]:
        for day in days(a.days):
            for n in [int(v) for v in a.n.split(',')]:
                print(f'{time.strftime("%m-%d %H:%M")}  n={n} |Omega|={scen} day {day} '
                      f'r={radius(scen):.4g}', flush=True)
                row = run_one(n, scen, day, a)
                print('  skipped (done)' if row is None else f'  {row}', flush=True)
    print('QUEUE DONE', flush=True)


if __name__ == '__main__':
    main()
