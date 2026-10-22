r"""
Generate ieee_owen/config_tables.tex — the setup subsection: prose AND its tables.

ONE FILE ON PURPOSE. The tables and the text that explains them belong together, the
way the Applied Energy manuscript had them: a reader meets the parameter choice and
the number in the same place, and a parameter that changes cannot leave its
explanation behind in another file. \input this in place of \subsection{Setup}.

Generated rather than hand-written on purpose. The two tables already in the manuscript
were transcribed by hand and the n=30 column drifted off its data (v^MIP, omega^LR and
both timings ended up carrying values from a single instance while the rest of the
column came from the 31-day sweep). A membership table is far easier to mistype than a
number table and far harder to catch, so the asset matrices below are read out of the
live CONFIGURATION_* objects and the shares are counted, never typed.

The scalar parameters ARE typed here, with the data_generator.py line they come from, on
the grounds that they are fixed constants of the market design rather than things that
move per experiment. Anything that varies across community size is derived.

Usage:  python ieee_owen/config_tables.py [-o ieee_owen/config_tables.tex]
        python ieee_owen/config_tables.py --supplement   # Tables S1-S3, online appendix
"""
import os
import sys
import argparse

_PAPER = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_PAPER)
sys.path.insert(0, _PAPER)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import large_community as LC

# 6-prosumer configuration, mirroring run_multiday.C6 (the archetype set every larger
# community is built out of). Kept here rather than imported so that generating the
# tables does not drag in the solver stack.
CONFIGURATION_6 = {
    'players_with_wind': ['u1'], 'players_with_solar': [],
    'players_with_electrolyzers': ['u2'], 'players_with_heatpumps': ['u3'],
    # u2 and u3 hold NO hydrogen / heat storage. They did once, but the 6-prosumer
    # parameter set carries storage_capacity_ratio_G = ratio_H = 0.0 (run_multiday.BASE6),
    # so those units had zero energy capacity and never moved -- the tables reported an
    # asset the dispatch could not use. Removing them changes no coalition value
    # (verified: v(N) identical to 1e-12 on days 1, 9 and 17); the larger communities set
    # the ratios to 3.0 and their hydrogen/heat storage IS active, which is why the counts
    # are nonzero from n=15 on.
    'players_with_elec_storage': ['u1'], 'players_with_hydro_storage': [],
    'players_with_heat_storage': [],
    'players_with_nfl_elec_demand': ['u4'], 'players_with_nfl_hydro_demand': ['u5'],
    'players_with_nfl_heat_demand': ['u6'],
}
PLAYERS_6 = [f'u{i}' for i in range(1, 7)]

# (config key, column head, short tag used in the role strings)
ASSETS = [
    ('players_with_wind',             'Wind',        'W'),
    ('players_with_solar',            'Solar',       'S'),
    ('players_with_electrolyzers',    'ELZ',         'E'),
    ('players_with_heatpumps',        'HP',          'H'),
    ('players_with_elec_storage',     r'E-sto',      'e'),
    ('players_with_hydro_storage',    r'H$_2$-sto',  'h'),
    ('players_with_heat_storage',     r'Q-sto',      'q'),
    ('players_with_nfl_elec_demand',  r'$D^{E}$',    'de'),
    ('players_with_nfl_hydro_demand', r'$D^{H_2}$',  'dh'),
    ('players_with_nfl_heat_demand',  r'$D^{Q}$',    'dq'),
]
# Assets that put first-stage commitment binaries in a prosumer's subproblem. Since
# 2026-09-30 only the electrolyzer is committed (compact_utility.hp_commitment defaults
# to False, so the heat pump is linear). The recourse binaries z^dir, z^sto and the
# segment selectors also enter every subproblem, so these members are not the only
# source of the duality gap; the row counts first-stage commitment, nothing more.
BINARY_ASSETS = ('players_with_electrolyzers',)

SIZES = [(6, PLAYERS_6, CONFIGURATION_6),
         (15, LC.PLAYERS_15, LC.CONFIGURATION_15),
         (30, LC.PLAYERS_30, LC.CONFIGURATION_30),
         (60, LC.PLAYERS_60, LC.CONFIGURATION_60)]


def has(cfg, key, p):
    return p in (cfg.get(key) or [])


def n_binary(cfg, players):
    """Members carrying at least one committed unit (union over BINARY_ASSETS)."""
    s = set()
    for k in BINARY_ASSETS:
        s |= set(cfg.get(k) or [])
    return len(s & set(players))


# --------------------------------------------------------------------------- prose
def prose_setup():
    return r"""\subsection{Setup}
\label{sec:setup}
The community of Section~\ref{sec:model} is instantiated at $|N|\in\{6,15,30,60\}$ over
$|T|=24$ hourly periods, with Korean (Jeju) load and price data and DK2 wind. Both
community-level channels are active: a symmetric reserve product sold hour by hour as
in the Nordic FCR-N market, paid hourly prices with the profile of the DK2 FCR-N
prices of January 2025 scaled to a daily mean of $56$~\euro/MW$\cdot$h, and a coincident-peak charge of
$150$~\euro/MW, so $m=(|K|+3)|T|=144$ at every size. Each size is run over $31$ daily
instances; aggregates are geometric means, which is the appropriate summary for a
quantity whose scaling in $n$ is the object of interest. The dispatch MILP, the
column-generation master and all pricing and separation problems are solved with SCIP,
the row-generation separation \eqref{eq:rg_sep} with Gurobi under a $3600$~s budget.
Timings are wall-clock with one solve running at a time.

Table~\ref{tab:params} collects the parameters and Table~\ref{tab:composition} the
composition of the four communities. Every parameter in Table~\ref{tab:params} is held
fixed across sizes, so that the size axis varies $n$ and nothing else.
"""


def prose_parameters():
    return r"""The electrolyzer carries the commitment binaries, a source of the
integrality gap $\omega^{\mathrm{LR}}$ that the proposed allocation gives up. It is a
three-state unit: besides on and off it holds a standby state drawing
$1\%$ of its rating, which is cheaper than a cold restart but not free, and returning
from off costs $50$~\euro. Its minimum load is $15\%$ of capacity, within the
$10$--$16\%$ reported for PEM units; the standby draw sits below the $2.5$--$5\%$ that
same literature reports, so the unit is modelled as somewhat cheaper to park than a
typical installation, which if anything understates the value of commitment and with it
$\omega^{\mathrm{LR}}$. The power-to-hydrogen curve is concave and is represented
exactly by six segments with segment-selection binaries. The heat pump enters linearly,
with no commitment state and no start-up charge: its heat output ranges continuously
between zero and $80\%$ of its rating at a constant coefficient of performance of
$3.28$, within ramp limits, and its reserve headroom is that window, measured in
electric MW while the window is on heat output.

Storage is sized relative to its owner rather than absolutely: rated power is a quarter
of the owner's asset capacity and energy capacity three times that power, so a member's
storage scales with the assets it actually operates. Demand is that of $700$ households,
and the community is a price taker facing an import price $1.5$ times its export price.
"""


def _binary_shares():
    return ', '.join(f'${100 * n_binary(cfg, pl) / n:.1f}\\%$' for n, pl, cfg in SIZES)


def prose_composition():
    return (r"""The four communities are nested: u1--u6 is a sub-configuration of u1--u15, that of
u1--u30, and that of u1--u60. A size comparison therefore adds members to a fixed
community rather than substituting one instance for another. At $n=6$ each function is
held by exactly one archetype---wind with electric storage, an electrolyzer with
hydrogen storage, a heat pump with heat storage, and one non-flexible consumer per
carrier---so every coalition value is traceable to a named role. The larger communities
introduce competition within each role, vertically integrated members owning generation
and conversion together, standalone storage operators, and consumers spanning two
carriers; Tables~\ref{tab:config15}--\ref{tab:config60} give the memberships.

The $n=60$ community doubles every count at $n=30$ exactly (Table~\ref{tab:composition}),
so those two sizes differ in $n$ alone. Its members are nonetheless not copies of the
ones they mirror: each carries its own capacity multiplier, listed in
Table~\ref{tab:config60}. The distinction is not cosmetic. A two-fold replica would have
a core that shrinks toward the competitive allocation by the Debreu--Scarf theorem, and
the measured behaviour across sizes would then be an artefact of replication rather than
of community size.

The last row of Table~\ref{tab:composition} counts the members carrying commitment
binaries (""" + _binary_shares() + r"""), so the share of committed units does not drift
with $n$.
""")


# --------------------------------------------------------------------------- tables
def tab_parameters():
    """Scalar parameters. Values from data_generator.py setup_lem_parameters()."""
    rows = [
        (r'\multicolumn{4}{@{}l}{\emph{Market and horizon}}', '', '', ''),
        (r'\quad Horizon $|T|$', '24', 'h', 'hourly resolution'),
        (r'\quad Carriers $|K|$', '3', '--', r'electricity, H$_2$, heat'),
        # (2|K|+3)|T|, not (|K|+3)|T|: the manuscript stacks the coupling rows in
        # <= form, so each of the |K| carrier balances is a PAIR of inequalities (draft
        # sec. III-B, and the m of Proposition `prop:eps`). The master in solver.py keeps
        # them as |K| equalities, which is the same system written differently -- but the
        # number quoted here has to be the manuscript's, or the O(|T|/n) bound is quoted
        # against a different m than it was proved for.
        (r'\quad Coupling rows $m=(2|K|+3)|T|$', '216', '--', 'fixed by the market design'),
        (r'\quad Households', '700', '--', 'demand scaling of the community'),
        (r'\quad Import/export price ratio', '1.5', '--', r'$\pi^{\mathrm{imp}}/\pi^{\mathrm{exp}}$'),
        (r'\quad H$_2$ reference price', '3.33', r'\euro/kg', '5000/1500'),
        (r'\quad Price data', r'\multicolumn{3}{l}{Jeju (KR) and DK2, January}', '', ''),
        (r'\midrule', '', '', ''),
        (r'\multicolumn{4}{@{}l}{\emph{Electrolyser (3-state commitment)}}', '', '', ''),
        (r'\quad Capacity $\overline{c}^{G}$', '1.0', 'MW', ''),
        (r'\quad Minimum load $c^{G}_{\min}$', '0.15', 'MW', r'$0.15\times$ capacity'),
        (r'\quad Standby draw $c^{G}_{\mathrm{sb}}$', '0.01', 'MW', ''),
        (r'\quad Start-up cost $c^{G}_{\mathrm{su}}$', '50', r'\euro', ''),
        (r'\quad Minimum down time', '2', 'h', ''),
        (r'\quad Efficiency curve', r'\multicolumn{3}{l}{piecewise linear, 6 segments}', '', ''),
        (r'\midrule', '', '', ''),
        (r'\multicolumn{4}{@{}l}{\emph{Heat pump (linear, no commitment)}}', '', '', ''),
        (r'\quad Capacity $\overline{c}^{H}$', '0.8', r'MW$_{\mathrm{th}}$', ''),
        (r'\quad Coefficient of performance $\nu^{\mathrm{cop}}$', '3.28', '--',
         'converts electric to heat'),
        (r'\quad Operating band $[0,c^{H}_{\max}]$', '[0, 0.8]', '--',
         'fraction of capacity'),
        (r'\quad Ramp limits', '0.9', '--', 'up/down, fraction of capacity per hour'),
        (r'\midrule', '', '', ''),
        (r'\multicolumn{4}{@{}l}{\emph{Storage (per carrier $k$)}}', '', '', ''),
        (r'\quad Power rating', '0.25', '--', r'$\times$ the owner\'s asset capacity'),
        (r'\quad Energy capacity', '3.0', '--', r'$\times$ power rating'),
        (r'\quad Initial state of charge', '0.2', '--', 'fraction of energy capacity'),
        # Charging and discharging efficiency, each way: nu_ch_k = nu_dis_k. Until
        # 2026-10-01 the electricity and hydrogen SOC balance read an unset key and ran
        # at 0.9 (this row said 0.90 for every carrier); it now reads nu_ch_E / nu_ch_G.
        (r'\quad Charging/discharging efficiency', r'0.95 / 0.90', '--',
         r'each way; electricity and H$_2$ / heat'),
        (r'\midrule', '', '', ''),
        (r'\multicolumn{4}{@{}l}{\emph{Reserve and peak coupling}}', '', '', ''),
        (r'\quad Reserve price $\pi^{\mathrm{res}}_t$', 'hourly', r'\euro/MW$\cdot$h',
         r'FCR-N DK2 Jan.\ 2025 profile, daily mean 56'),
        (r'\quad Reserve product', r'\multicolumn{3}{l}{symmetric, hourly (Nordic FCR-N)}', '', ''),
        (r'\quad Peak penalty $\delta^{\mathrm{peak}}$', '150', r'\euro/MW', ''),
    ]
    out = [r'% ---- generated by ieee_owen/config_tables.py ----',
           r'\begin{table}[!t]',
           r'\caption{Parameters of the community dispatch model. Held fixed across every '
           r'community size, so that the size axis varies $n$ alone.}',
           r'\label{tab:params}', r'\centering', r'\small',
           r'\begin{tabular}{@{}llll@{}}', r'\toprule',
           r'Parameter & Value & Unit & Note \\', r'\midrule']
    for r in rows:
        if r[0] == r'\midrule':
            out.append(r'\midrule')
        elif r[0].startswith(r'\multicolumn') or r'\multicolumn' in r[1]:
            out.append((r[0] if r[0].startswith(r'\multicolumn')
                        else f'{r[0]} & {r[1]}') + r' \\')
        else:
            out.append(f'{r[0]} & {r[1]} & {r[2]} & {r[3]} \\\\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(out)


def tab_composition():
    """Asset counts and shares across the four community sizes."""
    out = [r'% ---- generated by ieee_owen/config_tables.py ----',
           r'\begin{table}[!t]',
           r'\caption{Composition of the four communities. Counts are members owning the '
           r'asset, shares in brackets. The $n=60$ community doubles every count at '
           r'$n=30$ exactly, so those two sizes differ in $n$ and in nothing else; the '
           r'smaller communities are not similar to them in every share, the asset mix '
           r'having been set by role rather than by proportion at $n=6$ and $n=15$. What '
           r'is held across all four is the last row, the members carrying commitment '
           r'binaries, so the share of committed units does not drift with $n$.}',
           r'\label{tab:composition}', r'\centering', r'\small',
           r'\begin{tabular}{@{}l' + 'c' * len(SIZES) + r'@{}}', r'\toprule',
           ' & ' + ' & '.join(f'$n={n}$' for n, _, _ in SIZES) + r' \\', r'\midrule']
    for key, head, _ in ASSETS:
        cells = []
        for n, players, cfg in SIZES:
            c = len(set(cfg.get(key) or []) & set(players))
            cells.append(f'{c} ({100*c/n:.1f}\\%)' if c else '--')
        out.append(f'{head} & ' + ' & '.join(cells) + r' \\')
    out.append(r'\midrule')
    cells = []
    for n, players, cfg in SIZES:
        b = n_binary(cfg, players)
        cells.append(f'{b} ({100*b/n:.1f}\\%)')
    out.append(r'Members with commitment binaries & ' + ' & '.join(cells) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(out)


def tab_matrix(n, players, cfg, label, caption, mult=None):
    """Per-member asset matrix for one community size."""
    heads = ' & '.join(h for _, h, _ in ASSETS)
    ncol = len(ASSETS) + (2 if mult else 1)
    out = [r'% ---- generated by ieee_owen/config_tables.py ----',
           r'\begin{table}[!t]', f'\\caption{{{caption}}}', f'\\label{{{label}}}',
           r'\centering', r'\footnotesize',
           r'\setlength{\tabcolsep}{3pt}',
           r'\begin{tabular}{@{}l' + 'c' * (ncol - 1) + r'@{}}', r'\toprule',
           'Member & ' + heads + (r' & Cap.\ mult.' if mult else '') + r' \\',
           r'\midrule']
    for p in players:
        marks = [(r'$\bullet$' if has(cfg, k, p) else '') for k, _, _ in ASSETS]
        row = f'{p} & ' + ' & '.join(marks)
        if mult:
            # 'h2' and 'heat' are dropped: apply_60player_overrides scales
            # storage_{power,capacity}_{G,H}_<player>, and setup_lem_parameters creates
            # those keys for electricity only -- hydrogen and heat storage are sized once
            # for the whole community. The guard in the override makes the scaling a
            # silent no-op, so printing it would report a differentiation that the
            # dispatch never sees.
            m = {k: v for k, v in (mult.get(p) or {}).items() if k not in ('h2', 'heat')}
            row += ' & ' + (', '.join(f'{k} {v}' for k, v in m.items()) if m else '1.0')
        out.append(row + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    return '\n'.join(out)


# ------------------------------------------------------- online appendix (TEMPR)
# Tables S1-S3 of the online appendix (supplementary_tempr.tex on Overleaf), in the
# compact format and the carrier notation of that file: K = {E, G, H}, G hydrogen and
# H heat, as in the manuscript. `--supplement` writes them to supplement_tables.tex,
# whose three blocks are pasted into the appendix in place of the old ones.
SUP_ASSETS = [
    ('players_with_wind',             'W'),
    ('players_with_solar',            'S'),
    ('players_with_electrolyzers',    'EZ'),
    ('players_with_heatpumps',        'HP'),
    ('players_with_elec_storage',     r'B$_{E}$'),
    ('players_with_hydro_storage',    r'B$_{G}$'),
    ('players_with_heat_storage',     r'B$_{H}$'),
    ('players_with_nfl_elec_demand',  r'D$_{E}$'),
    ('players_with_nfl_hydro_demand', r'D$_{G}$'),
    ('players_with_nfl_heat_demand',  r'D$_{H}$'),
]
# _MULT_60 tags printed in the right panel of Table S3. 'h2' and 'heat' are left out:
# hydrogen and heat storage are sized by the community-wide keys storage_power_G/H
# (data_generator.setup_lem_parameters), so apply_60player_overrides finds no
# per-member key to scale and those multipliers never reach the dispatch.
SUP_MULT = (('renewable', 'R'), ('els', 'EZ'), ('hp', 'HP'), ('ess', r'B$_{E}$'))

# Heat pump, data_generator: hp_cap = 0.8 MW of heat, c_max_H = 0.8, nu_cop = 3.28.
# The manuscript bounds the electric input, d^H_sc <= d-bar^H, so the table reports
# d-bar^H = c_max_H * hp_cap / nu_cop.
_HP_CAP, _HP_CMAX, _HP_COP = 0.8, 0.8, 3.28


def sup_tab_params():
    d_h = _HP_CMAX * _HP_CAP / _HP_COP
    g = r'\grouphead{%s} \\'
    r = r'\hspace{0.6em}%s & %s & %s \\'
    rows = [
        g % 'Market and horizon',
        r % ('Horizon $|T|$', r'24\,h', 'hourly resolution'),
        r % ('Households', '700', ''),
        r % ('Import price markup', '0.5',
             r'$(\pi_t^{k,imkt}-\pi_t^{k,emkt})/|\pi_t^{k,emkt}|$'),
        r % (r'H$_2$ reference price', r'3.33\,\euro/kg', ''),
        r'\hspace{0.6em}Load and price data & \multicolumn{2}{l}{Jeju (KR), January} \\',
        r'\hspace{0.6em}Wind data & \multicolumn{2}{l}{DK2, January} \\',
        r'\addlinespace[1pt]',
        g % 'Electrolyzer (three-state commitment)',
        r % (r'Rated draw $\overline d^{G}$', r'1.0\,MW', ''),
        r % (r'Minimum draw $\underline d^{G}$', r'0.15\,MW', r'$0.15\times$ rated'),
        r % (r'Standby draw $c_G^{sb}$', r'0.01\,MW', ''),
        r % (r'Start-up cost $c_j^{G,su}$', r'50\,\euro', ''),
        r % ('Minimum down time', r'2\,h', ''),
        r'\hspace{0.6em}Efficiency curve & \multicolumn{2}{l}{piecewise linear, 6 segments} \\',
        r'\addlinespace[1pt]',
        g % 'Heat pump (linear, no commitment)',
        r % (r'Rated input $\overline d^{H}$', rf'{d_h:.3f}\,MW',
             rf'{_HP_CMAX * _HP_CAP:.2f}\,MW$_{{\mathrm{{th}}}}$ of heat'),
        r % (r'Coeff.\ of performance $\nu^{\mathrm{COP}}$', f'{_HP_COP}', r'el.\ to heat'),
        r'\addlinespace[1pt]',
        g % 'Storage (per carrier $k$)',
        r % ('Power rating', '0.25', r'$\times$ asset capacity'),
        r % ('Energy capacity', '3.0', r'$\times$ power rating'),
        r % ('Initial state of charge', '0.2', 'of energy capacity'),
        r % (r'Ch./dis.\ efficiency, E, H$_2$', '0.95', 'each way'),
        r % (r'Ch./dis.\ efficiency, heat', '0.90', 'each way'),
        r'\addlinespace[1pt]',
        g % 'Reserve and peak coupling',
        r % (r'Reserve price $\pi_t^{res}$', 'hourly', r'DK2 FCR-N, Jan.\ 2025'),
        r % (r'Daily mean of $\pi_t^{res}$', r'56\,\euro/MW$\cdot$h', 'rescaled'),
        r'\hspace{0.6em}Reserve product & \multicolumn{2}{l}{symmetric, $|I|=24$ blocks of 1\,h} \\',
        r % (r'Shortfall penalty factor $\kappa$', '5', ''),
        r % (r'Peak penalty $\delta^{peak}$', r'150\,\euro/MW', ''),
    ]
    return '\n'.join([
        r'% ---- generated by ieee_owen/config_tables.py --supplement ----',
        r'\begin{table}[!t]',
        r'\caption{Parameters of the EC dispatch model. Held fixed across every '
        r'community size, so that the size axis varies $n$ alone.}',
        r'\label{tab:params}', r'\centering', r'\footnotesize',
        r'\setlength{\tabcolsep}{2pt}', r'\renewcommand{\arraystretch}{1.05}',
        r'\begin{tabular}{@{}l r@{\hspace{6pt}} l@{}}', r'\toprule',
        r'Parameter & Value & Note \\', r'\midrule',
        *rows, r'\bottomrule', r'\end{tabular}', r'\end{table}'])


def sup_tab_composition():
    out = [r'% ---- generated by ieee_owen/config_tables.py --supplement ----',
           r'\begin{table}[!t]',
           r'\caption{Composition of the four communities. Entries are the number of '
           r'members owning the asset, with the share of the community in brackets, in '
           r'percent. W wind, S solar, EZ electrolyzer, HP heat pump, '
           r'B$_{E}$/B$_{G}$/B$_{H}$ storage and D$_{E}$/D$_{G}$/D$_{H}$ demand of '
           r'electricity, hydrogen and heat. The last row counts the members whose '
           r'electrolyzer commitment is a first-stage decision.}',
           r'\label{tab:composition}', r'\centering', r'\footnotesize',
           r'\setlength{\tabcolsep}{3pt}',
           r'\begin{tabular}{@{}l' + ' r' * len(SIZES) + r'@{}}', r'\toprule',
           ' & ' + ' & '.join(f'$n={n}$' for n, _, _ in SIZES) + r' \\', r'\midrule']

    def cell(c, n):
        return f'{c} ({100 * c / n:.1f})' if c else '--'
    for key, head in SUP_ASSETS:
        out.append(f'{head} & ' + ' & '.join(
            cell(len(set(cfg.get(key) or []) & set(pl)), n) for n, pl, cfg in SIZES) + r' \\')
    out.append(r'\midrule')
    out.append('Commitment & ' + ' & '.join(
        cell(n_binary(cfg, pl), n) for n, pl, cfg in SIZES) + r' \\')
    out += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(out)


def _sup_rows(players, cfg, mult=None, rules_after=()):
    rows = []
    for p in players:
        marks = [r'\dt' if has(cfg, k, p) else '' for k, _ in SUP_ASSETS]
        row = f'{p} & ' + ' & '.join(marks)
        if mult is not None:
            m = mult.get(p) or {}
            row += ' & ' + ', '.join(f'{lab}{m[tag]}' for tag, lab in SUP_MULT if tag in m)
        rows.append(row.rstrip() + r' \\')
        if p in rules_after:
            rows.append(r'\midrule')
    return rows


def sup_tab_config():
    heads = ' & '.join(h for _, h in SUP_ASSETS)
    left = [p for p in LC.PLAYERS_30]
    right = [p for p in LC.PLAYERS_60 if int(p[1:]) > 30]
    ncol = len(SUP_ASSETS)
    out = [r'% ---- generated by ieee_owen/config_tables.py --supplement ----',
           r'\begin{table*}[!t]',
           r'\caption{Member configuration of the four communities. Left panel, u1--u30; '
           r'right panel, the members added at $n=60$, with the capacity multiplier '
           r'applied to each (\texttt{apply\_60player\_overrides}); a blank multiplier '
           r'means $1.0$. Horizontal rules in the left panel mark the $n=6$ and $n=15$ '
           r'communities. Columns as in Table~\ref{tab:composition}. Hydrogen and heat '
           r'storage have the same size for every owner, so the multipliers scale '
           r'renewables (R), conversion units and electricity storage only.}',
           r'\label{tab:config}', r'\centering', r'\scriptsize',
           r'\setlength{\tabcolsep}{3pt}',
           r'\begin{tabular}[t]{@{}l' + 'c' * ncol + r'@{}}', r'\toprule',
           f'Member & {heads} \\\\', r'\midrule',
           *_sup_rows(left, LC.CONFIGURATION_30, rules_after=('u6', 'u15')),
           r'\bottomrule', r'\end{tabular}', r'\hspace{2.5em}',
           r'\begin{tabular}[t]{@{}l' + 'c' * ncol + r'l@{}}', r'\toprule',
           f'Member & {heads} & Cap.\\ mult. \\\\', r'\midrule',
           *_sup_rows(right, LC.CONFIGURATION_60, mult=LC._MULT_60),
           r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-o', '--out', default=os.path.join(_PAPER, 'config_tables.tex'))
    ap.add_argument('--supplement', nargs='?', const=os.path.join(_PAPER, 'supplement_tables.tex'),
                    default=None, help='write Tables S1-S3 of the online appendix instead '
                                       '(default ieee_owen/supplement_tables.tex)')
    args = ap.parse_args()

    if args.supplement:
        with open(args.supplement, 'w') as f:
            f.write('\n\n'.join([sup_tab_params(), sup_tab_composition(),
                                 sup_tab_config()]) + '\n')
        print(f'wrote {args.supplement}')
        return

    parts = [
        '% =====================================================================',
        '% Setup subsection: the prose and the tables it explains, in one file.',
        '% GENERATED by ieee_owen/config_tables.py -- edit that, not this file.',
        '% The asset matrices are read out of large_community.CONFIGURATION_* and the',
        '% shares are counted, so this file cannot drift away from what was solved.',
        '% \\input this in place of \\subsection{Setup} in ieee_draft.txt.',
        '% Requires: booktabs, eurosym (\\euro).',
        '% =====================================================================',
        '',
        prose_setup(),
        tab_parameters(),
        prose_parameters(),
        prose_composition(),
        tab_composition(),
        tab_matrix(15, LC.PLAYERS_15, LC.CONFIGURATION_15, 'tab:config15',
                   'Member configuration of the 15-prosumer community. '
                   'u1--u6 are the 6-prosumer community unchanged.'),
        tab_matrix(30, [p for p in LC.PLAYERS_30 if int(p[1:]) > 15],
                   LC.CONFIGURATION_30, 'tab:config30',
                   'Members added at $n=30$. u1--u15 are Table~\\ref{tab:config15} '
                   'unchanged, so the 15-prosumer community is a sub-configuration of '
                   'this one.'),
        tab_matrix(60, [p for p in LC.PLAYERS_60 if int(p[1:]) > 30],
                   LC.CONFIGURATION_60, 'tab:config60',
                   'Members added at $n=60$. u1--u30 are unchanged. The last column is '
                   'the capacity multiplier applied to that member '
                   '(\\texttt{apply\\_60player\\_overrides}); every added member differs '
                   'from the archetype it mirrors, so the community is not a two-fold '
                   'replica of the 30-prosumer one -- a replica would shrink the core '
                   'toward the competitive allocation by Debreu--Scarf and the size '
                   'comparison would measure that instead.',
                   mult=LC._MULT_60),
    ]
    with open(args.out, 'w') as f:
        f.write('\n'.join(parts))
    print(f'wrote {args.out}')
    for n, players, cfg in SIZES:
        print(f'  n={n:2d}: {n_binary(cfg, players)} members with commitment binaries '
              f'({100*n_binary(cfg, players)/n:.1f}%)')


if __name__ == '__main__':
    main()
