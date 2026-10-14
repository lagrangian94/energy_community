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

On Linux the memory comes from /proc: peak_ws is the kernel's peak resident set
(VmHWM) and peak_private the largest anonymous + swapped size seen at a poll, so a
spike between two polls is in the first and not in the second.

    python ieee_owen/weak_eps_experiment/run_kl_main.py                 # whole queue
    python ieee_owen/weak_eps_experiment/run_kl_main.py --scen 20 --n 6,15 --days 1-3
"""
import os, sys, time, shutil, argparse, subprocess

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
os.chdir(_ROOT)

from scipy.stats import chi2

OUT = os.path.join('ieee_owen', 'weak_eps_experiment', 'stochastic', 'main17_xi')
PRICE_SCALE = 17.0 / 56.0
ALPHA = 0.05
POLL = 5
# Linux: every job of every queue runs in one systemd user slice whose memory is capped
# below the machine's (swap excluded), so when the jobs together run out the kernel kills
# one of them, not the machine. (The two crashes of cykim on 2026-10-14 were not this:
# their kernel reports show corrupted kernel memory, no out-of-memory; see the plan,
# sec. 8.20 (4).) EC_JOBS_MEM_GB sets the cap (default:
# installed memory less 16 GB), EC_JOBS_SLICE=0 turns the slice off. A job also waits to
# start until START_FREE GB are available.
SLICE = 'ecjobs.slice'
START_FREE = 16.0
_slice = None                                # None: not set up yet; False: unavailable

if os.name == 'nt':
    import ctypes, ctypes.wintypes as wt

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


    def probe(p):
        """(pid, handle) of the interpreter behind the launcher p, once it is there."""
        kid = child_of(p.pid)
        return (kid, k32.OpenProcess(0x1410, False, kid)) if kid else (None, None)


    def peaks(handle):
        c = mem(handle)
        return c.PeakPagefileUsage / 2**30, c.PeakWorkingSetSize / 2**30


    def kill(p, kid):
        p.kill()
        if kid:
            subprocess.run(['taskkill', '/F', '/T', '/PID', str(kid)], capture_output=True)


else:
    def avail_gb():
        for ln in open('/proc/meminfo'):
            if ln.startswith('MemAvailable:'):
                return int(ln.split()[1]) / 2**20
        return float('inf')


    def probe(p):
        return p.pid, p.pid                  # no launcher: p is the interpreter


    def peaks(pid):
        """(anonymous + swapped now, peak resident) [GB]; zeros once the process is gone."""
        kb = {}
        try:
            for ln in open(f'/proc/{pid}/status'):
                key, _, val = ln.partition(':')
                if key in ('RssAnon', 'VmSwap', 'VmHWM'):
                    kb[key] = int(val.split()[0])
        except OSError:
            pass
        return ((kb.get('RssAnon', 0) + kb.get('VmSwap', 0)) / 2**20,
                kb.get('VmHWM', 0) / 2**20)


    def kill(p, kid):
        p.kill()                             # pricing workers are threads: no tree


def radius(scen):
    return float(chi2.ppf(1 - ALPHA, 1) / (2 * scen))


def command(n, scen, day, out, ef_round_limit=10800, nodefile=3.0, ef_log=None):
    """The stochastic_extension.py call of one run of the queue."""
    tag = f'n{n}_S{scen}_day{day}'
    return [sys.executable, os.path.join('ieee_owen', 'stochastic_extension.py'),
            '--n', str(n), '--scenarios', str(scen), '--day', str(day),
            '--kl-radius', repr(radius(scen)), '--kl-master', 'dual',
            '--reserve-price-scale', repr(PRICE_SCALE), '--stall-barrier',
            '--mip-time-limit', str(ef_round_limit), '--ef-nodefile-start', str(nodefile),
            '--ef-cache', os.path.join(out, 'ef_cache', f'{tag}.pkl'),
            '--ef-log', ef_log or os.path.join(out, 'logs', f'{tag}_ef_gurobi.log'),
            '--tag', 'main', '--out', out]


def in_slice(cmd):
    """cmd inside the capped slice (Linux with a user systemd), else cmd unchanged."""
    global _slice
    if os.name == 'nt' or os.environ.get('EC_JOBS_SLICE') == '0' or _slice is False:
        return cmd
    if _slice is None:
        total = next(int(ln.split()[1]) for ln in open('/proc/meminfo')
                     if ln.startswith('MemTotal:')) / 2**20
        cap = float(os.environ.get('EC_JOBS_MEM_GB', total - 16.0))
        ok = shutil.which('systemd-run') and subprocess.run(
            ['systemctl', '--user', 'set-property', '--runtime', SLICE,
             f'MemoryMax={int(cap)}G', 'MemorySwapMax=0'], capture_output=True).returncode == 0
        _slice = bool(ok)
        print(f'jobs capped at {int(cap)} GB together ({SLICE})' if ok else
              'no user systemd: jobs run without the memory cap', flush=True)
        if not ok:
            return cmd
    # --scope runs cmd in this process tree, so p.pid stays the interpreter's
    return ['systemd-run', '--user', '--scope', '--quiet', f'--slice={SLICE}', '--'] + cmd


def watched(cmd, log, min_free_gb, kill_after=None):
    """Run cmd with its output in log, polling the interpreter's memory; kill it if
    the machine's available memory falls below min_free_gb, or after kill_after
    seconds. Returns (exit, wall_s, peak_private_gb, peak_ws_gb, min_avail_gb, killed).
    On Linux the job runs inside the capped slice (in_slice) and starts only once
    START_FREE GB are available."""
    os.makedirs(os.path.dirname(log), exist_ok=True)
    cmd = in_slice(cmd)
    if os.name != 'nt' and avail_gb() < START_FREE:
        print(f'  waiting for {START_FREE:.0f} GB of free memory '
              f'({avail_gb():.1f} GB now)', flush=True)
        while avail_gb() < START_FREE:
            time.sleep(10)
    env = dict(os.environ, PYTHONHASHSEED='21', PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8')
    # the dispatch model saves its plots from whatever thread builds it; with a display
    # matplotlib picks Tk, which aborts the interpreter when a worker thread drops a figure
    env.setdefault('MPLBACKEND', 'Agg')
    t0 = time.time()
    with open(log, 'w', encoding='utf-8') as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
        kid, h = None, None
        peak_priv = peak_ws = 0.0
        min_free, killed = float('inf'), False
        while p.poll() is None:
            if h is None:                    # the interpreter may not be spawned yet
                kid, h = probe(p)
            if h:
                priv, ws = peaks(h)
                peak_priv, peak_ws = max(peak_priv, priv), max(peak_ws, ws)
            free = avail_gb()
            min_free = min(min_free, free)
            if free < min_free_gb or (kill_after and time.time() - t0 > kill_after):
                kill(p, kid)
                killed = True
                break
            time.sleep(POLL if h else 2)
        rc = p.wait()
    if rc == -9 and not killed and _slice:
        with open(log, 'a', encoding='utf-8') as f:
            f.write(f'\nkilled by the memory cap of {SLICE}\n')
    if h:
        priv, ws = peaks(h)
        peak_priv, peak_ws = max(peak_priv, priv), max(peak_ws, ws)
    return rc, time.time() - t0, peak_priv, peak_ws, min_free, killed


def run_one(n, scen, day, a):
    log = os.path.join(a.out, 'logs', f'n{n}_S{scen}_day{day}.log')
    if os.path.exists(log) and 'wrote ' in open(log, encoding='utf-8', errors='replace').read():
        return None
    rc, wall, peak_priv, peak_ws, min_free, killed = watched(
        command(n, scen, day, a.out, a.ef_round_limit, a.nodefile), log, a.min_free)
    row = (f'{time.strftime("%Y-%m-%d %H:%M")},{n},{scen},{day},{radius(scen):.6g},{rc},'
           f'{wall:.0f},{peak_priv:.2f},{peak_ws:.2f},{min_free:.2f},{int(killed)}')
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
