import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np

# ── Colors ──
C_IP   = '#2E86AB'   # Blue
C_CHP  = '#F18F01'   # Orange
C_OFF  = '#000000'   # Black
C_BP   = '#d4a017'   # Yellow
C_GAP  = '#c0392b'   # Red

# ── Parameters ──
P_MIN = 0.15
BP1 = 0.2823
a1, b1 = 21.12, -0.379
a2, b2 = 18.82, 0.270
DISPATCH = 0.260

seg1 = lambda p: a1 * p + b1
seg2 = lambda p: a2 * p + b2
chord_slope = seg1(BP1) / BP1
chord = lambda p: chord_slope * p

# ── Figure ──
fig, ax = plt.subplots(figsize=(6.5, 4.0))
fig.subplots_adjust(left=0.10, right=0.65, top=0.96, bottom=0.15)

# ── Convex hull fill (no legend) ──
p_fill = np.linspace(P_MIN, BP1, 100)
ax.fill_between(p_fill, seg1(p_fill), chord(p_fill),
                color=C_CHP, alpha=0.06, zorder=1)
p_tri_x = [0, P_MIN, P_MIN, 0]
p_tri_y = [0, chord(P_MIN), seg1(P_MIN), 0]
ax.fill(p_tri_x, p_tri_y, color=C_CHP, alpha=0.03, zorder=1)

# ── Segment 1 extension (dashed) ──
p_ext = np.linspace(0, P_MIN, 50)
ax.plot(p_ext, seg1(p_ext), color=C_IP, linewidth=1.0,
        linestyle='--', alpha=0.35, zorder=2)

# ── ON curve: segment 1 ──
p_s1 = np.linspace(P_MIN, BP1, 100)
ax.plot(p_s1, seg1(p_s1), color=C_IP, linewidth=2.2, zorder=3,
        label=r'IP: ON segment ($p^G = \phi_s^1 d^{E,sc,G} + \phi_s^0$)')

# ── ON curve: segment 2 (lighter) ──
p_s2 = np.linspace(BP1, 0.36, 60)
ax.plot(p_s2, seg2(p_s2), color=C_IP, linewidth=1.5, alpha=0.3, zorder=3)

# ── Chord ──
p_ch = np.linspace(0, BP1, 100)
ax.plot(p_ch, chord(p_ch), color=C_CHP, linewidth=2.2,
        linestyle=(0, (6, 3)), zorder=3,
        label=r'CHP: Chord (conv. hull boundary)')

# ── OFF point ──
# ax.plot(0, 0, 'o', color=C_OFF, markersize=5, zorder=5,
#         markeredgecolor='white', markeredgewidth=1.5)
ax.annotate(r'Off $(0,0)$', xy=(0, 0), xytext=(0.015, 0.25),
            fontsize=8, color=C_OFF, fontweight='bold')

# ── Negative intercept ──
# ax.plot(0, b1, 'x', color=C_IP, markersize=5, zorder=4, markeredgewidth=1.8)
ax.annotate(r'$\phi_1^0 = -0.379$', xy=(0, b1),

            xytext=(0.02, b1), fontsize=7.5, color=C_IP, alpha=0.7, va='center')
ax.axhline(y=b1, color=C_IP, linewidth=0.5, linestyle=':', alpha=0.25, zorder=1)

# ── Segment boundary ──
ax.plot(BP1, seg1(BP1), 'o', color=C_BP, markersize=7, zorder=5,
        markeredgecolor='white', markeredgewidth=1.5)
ax.annotate(r'$\bar{P}_1$',
            xy=(BP1, seg1(BP1)), xytext=(BP1 + 0.008, seg1(BP1) + 0.3),
            fontsize=8, color=C_BP, fontweight='bold')

# ── Dispatch points ──
y_ip = seg1(DISPATCH)
y_chp = chord(DISPATCH)
ax.plot([DISPATCH, DISPATCH], [0, y_chp + 0.1],
        color='#888', linewidth=0.7, linestyle=':', zorder=2)
ax.plot(DISPATCH, y_ip, 'o', color=C_IP, markersize=6, zorder=5,
        markeredgecolor='white', markeredgewidth=1.3)
ax.plot(DISPATCH, y_chp, 'o', color=C_CHP, markersize=6, zorder=5,
        markeredgecolor='white', markeredgewidth=1.3)

# (gap shown in inset)

# ── Main axes ──
ax.set_xlim(-0.02, 0.34)
ax.set_ylim(-0.75, 7.0)
ax.set_xlabel(r'Electricity input $d^{E,sc,G}$ (MW)', fontsize=9.5)
ax.set_ylabel(r'H$_2$ production $p^G$ (kg/h)', fontsize=9.5)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(0.7)
ax.spines['bottom'].set_linewidth(0.7)
ax.tick_params(labelsize=8)
ax.legend(fontsize=6.5, loc='upper left', framealpha=0.95,
          edgecolor='#ddd', fancybox=False)

# ══════════════════════════════════════════════════════════
# INSET in right margin, compact
# ══════════════════════════════════════════════════════════
axins = fig.add_axes([0.70, 0.30, 0.24, 0.42])

inset_arrow_len = 0.07
inset_xmin = DISPATCH - 0.003
inset_xmax = DISPATCH + inset_arrow_len + 0.003
inset_p = np.linspace(inset_xmin, inset_xmax, 200)

axins.plot(inset_p, seg1(inset_p), color=C_IP, linewidth=2.0, zorder=3)
axins.plot(inset_p, chord(inset_p), color=C_CHP, linewidth=2.0,
           linestyle=(0, (5, 2.5)), zorder=3)

# Fill gap
axins.fill_between(inset_p, seg1(inset_p), chord(inset_p),
                   color=C_GAP, alpha=0.08, zorder=1)

# Tip coordinates
tip_x = DISPATCH + inset_arrow_len
tip_ip_y = y_ip + a1 * inset_arrow_len
tip_chp_y = y_chp + chord_slope * inset_arrow_len

# Arrows
axins.annotate('', xy=(tip_x, tip_ip_y),
               xytext=(DISPATCH, y_ip),
               arrowprops=dict(arrowstyle='->', color=C_IP, lw=2.2,
                               shrinkA=0, shrinkB=0))
axins.annotate('', xy=(tip_x, tip_chp_y),
               xytext=(DISPATCH, y_chp),
               arrowprops=dict(arrowstyle='->', color=C_CHP, lw=2.2,
                               shrinkA=0, shrinkB=0))

# Points
axins.plot(DISPATCH, y_ip, 'o', color=C_IP, markersize=5, zorder=5,
           markeredgecolor='white', markeredgewidth=1.2)
axins.plot(DISPATCH, y_chp, 'o', color=C_CHP, markersize=5, zorder=5,
           markeredgecolor='white', markeredgewidth=1.2)

# Labels at midpoint of arrows
mid_x = DISPATCH + inset_arrow_len * 0.5
mid_ip_y = y_ip + a1 * inset_arrow_len * 0.5
mid_chp_y = y_chp + chord_slope * inset_arrow_len * 0.5
axins.text(mid_x, mid_ip_y + 0.06,
           r'IP: $\phi_1^1 = 21.12$',
           fontsize=7, color=C_IP, fontweight='bold', va='bottom', ha='center')
axins.text(mid_x, mid_chp_y - 0.06,
           r'CHP: $19.78$',
           fontsize=7, color=C_CHP, fontweight='bold', va='top', ha='center')

# Inset config
axins.set_xlim(inset_xmin, inset_xmax)
axins.set_ylim(min(y_ip, y_chp) - 0.08, max(tip_ip_y, tip_chp_y) + 0.08)
axins.set_xticks([])
axins.set_yticks([5.2, 6.6])
axins.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
axins.tick_params(labelsize=6, length=2)
axins.set_facecolor('white')
for sp in axins.spines.values():
    sp.set_linewidth(0.6)
    sp.set_edgecolor('#888')

# ── Zoom box on main plot ──
rect_x0, rect_x1 = DISPATCH - 0.012, DISPATCH + 0.012
rect_y0, rect_y1 = min(y_ip, y_chp) - 0.1, max(y_ip, y_chp) + 0.1
rect = plt.Rectangle((rect_x0, rect_y0), rect_x1 - rect_x0, rect_y1 - rect_y0,
                      linewidth=0.8, edgecolor='#888', facecolor='none', zorder=4)
ax.add_patch(rect)

con1 = ConnectionPatch(xyA=(rect_x1, rect_y1), coordsA=ax.transData,
                       xyB=(0, 1), coordsB=axins.transAxes,
                       color='#aaa', linewidth=0.6, linestyle='--')
con2 = ConnectionPatch(xyA=(rect_x1, rect_y0), coordsA=ax.transData,
                       xyB=(0, 0), coordsB=axins.transAxes,
                       color='#aaa', linewidth=0.6, linestyle='--')
fig.add_artist(con1)
fig.add_artist(con2)

# plt.tight_layout()
plt.savefig('/mnt/f/github_projects/energy_community/results_csuG0/convex_hull_geometry_pwl.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig('/mnt/f/github_projects/energy_community/results_csuG0/convex_hull_geometry_pwl.png',
            dpi=300, bbox_inches='tight')
print("Done.")