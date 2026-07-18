import matplotlib.colors as mcolors
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xrayutilities as xu
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.special import kv

from nrxrdct.laue.camera import Camera
from nrxrdct.laue.layers import LayeredCrystal

from .simulation import beam_in_crystal, synchrotron_spectrum

# from .simulation import (  # A_LATTICE,; HARMONIC_WIDTH,; N_HARMONICS,; PHI1_DEG,; PHI2_DEG,; PHI_DEG,; E_FUNDAMENTAL_eV,
#     beam_in_crystal,
#     synchrotron_spectrum,
# )

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

BG = "#080c14"
FG = "#ccccee"
COL_BCC = "#4fc3f7"
COL_SUP = "#ff6633"
COL_DB = "#ffffaa"


def _ax_style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, fontsize=9, pad=5)
    ax.tick_params(colors="#7788aa", labelsize=6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1f2e")


# ─────────────────────────────────────────────────────────────────────────────
# 2θ / χ  AND  GNOMONIC PROJECTION PLOTS
# ─────────────────────────────────────────────────────────────────────────────


def _uf_from_tth_chi(tth_deg, chi_deg):
    """Scattered unit vector from 2theta, chi  (LaueTools LT2 frame)."""
    tth = np.radians(tth_deg)
    chi = np.radians(chi_deg)
    return np.array(
        [-np.sin(tth) * np.sin(chi), np.cos(tth), np.sin(tth) * np.cos(chi)]
    )


def _gnomonic(tth_deg, chi_deg):
    """
    Gnomonic projection of a scattered beam onto the plane perpendicular
    to the forward beam direction (tangent plane at the north pole of the
    unit sphere).

        gX = -sin(2θ) sin χ  /  (1 + cos 2θ)
        gY =  sin(2θ) cos χ  /  (1 + cos 2θ)

    For 2θ < 90°  the point lies inside the unit circle (|g| < 1).
    For 2θ = 90°  |g| = 1.
    For 2θ > 90°  |g| > 1 (back-hemisphere).
    Straight lines in gnomonic space = crystallographic zones.
"""
    tth = np.asarray(tth_deg, float)
    chi = np.asarray(chi_deg, float)
    denom = 1.0 + np.cos(np.radians(tth))
    # guard against 2theta = 180 (denom = 0)
    safe = np.where(np.abs(denom) > 1e-10, denom, np.nan)
    gX = -np.sin(np.radians(tth)) * np.sin(np.radians(chi)) / safe
    gY = np.sin(np.radians(tth)) * np.cos(np.radians(chi)) / safe
    return gX, gY


def _style_angular_ax(ax, title):
    ax.set_facecolor("#080c14")
    ax.set_title(title, color="#ccccee", fontsize=9, pad=5)
    ax.tick_params(colors="#7788aa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1f2e")
    ax.grid(True, ls=":", lw=0.35, color="#181e2e")


def plot_2theta_chi(
    spots_bcc, spots_b2, E_MIN_eV=5_000, E_MAX_eV=27_000, out_path="laue_2theta_chi.png"
):
    """
    Plot Laue patterns in angular space: two representations side-by-side
    for each phase (BCC and B2):

    Left column  – 2θ vs χ scatter plot  (LaueTools .cor file convention)
    Right column – Gnomonic projection   (gX, gY)
                   Straight lines = crystallographic zone axes
                   Unit circle = 2θ = 90°

    Spots are coloured by photon energy and sized by normalised intensity.
    Fundamental reflections: circles (○).
    B2 superlattice reflections: stars (★) in orange.

    Args:
        spots_bcc, spots_b2 (lists of spot dicts from simulate_laue()):
        out_path (output PNG path):
"""
    import matplotlib.colors as mcolors
    import matplotlib.gridspec as mgridspec
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    BG = "#080c14"
    FG = "#ccccee"
    COL_FUND = "#4fc3f7"
    COL_SUPER = "#ff6633"

    all_E = [s["E"] for s in spots_bcc + spots_b2]
    E_norm = mcolors.Normalize(vmin=E_MIN_eV / 1e3, vmax=E_MAX_eV / 1e3)
    cmap = "plasma"

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor(BG)

    gs = mgridspec.GridSpec(
        2,
        3,
        width_ratios=[1, 1, 0.06],
        height_ratios=[1, 1],
        hspace=0.38,
        wspace=0.28,
        left=0.07,
        right=0.93,
        top=0.92,
        bottom=0.07,
    )

    ax_bcc_ang = fig.add_subplot(gs[0, 0])
    ax_bcc_gno = fig.add_subplot(gs[0, 1])
    ax_b2_ang = fig.add_subplot(gs[1, 0])
    ax_b2_gno = fig.add_subplot(gs[1, 1])
    ax_cb = fig.add_subplot(gs[:, 2])

    # ── helper: 2theta vs chi scatter ────────────────────────────────────────
    def draw_tth_chi(ax, spots, title):
        _style_angular_ax(ax, title)
        ax.set_ylabel("χ  (degrees)", color="#7788aa", fontsize=8)
        ax.set_xlabel("2θ  (degrees)", color="#7788aa", fontsize=8)
        ax.axvline(0, color="#252b40", lw=0.8)

        fund = [s for s in spots if not s.get("is_superlattice", False)]
        super_ = [s for s in spots if s.get("is_superlattice", False)]

        for subset, mk, ec in [(fund, "o", COL_FUND), (super_, "*", COL_SUPER)]:
            if not subset:
                continue
            chis = [s["chi"] for s in subset]
            tths = [s["tth"] for s in subset]
            Es = [s["E"] / 1e3 for s in subset]
            sz = [max(4, 80 * s["intensity"] ** 0.4) for s in subset]
            ax.scatter(
                tths,
                chis,
                s=sz,
                c=Es,
                cmap=cmap,
                norm=E_norm,
                alpha=0.80,
                edgecolors=ec,
                linewidths=0.35,
                marker=mk,
                zorder=3,
            )

        # Label 8 strongest fundamental spots
        for s in sorted(fund, key=lambda x: -x["intensity"])[:8]:
            h, k, l = s["hkl"]
            ax.annotate(
                f"({h}{k}{l})",
                xy=(s["tth"], s["chi"]),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=5.5,
                color="#aaccff",
                alpha=0.9,
            )

        # Label 4 strongest superlattice spots
        for s in sorted(super_, key=lambda x: -x["intensity"])[:4]:
            h, k, l = s["hkl"]
            ax.annotate(
                f"({h}{k}{l})",
                xy=(s["tth"], s["chi"]),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=5.5,
                color=COL_SUPER,
                alpha=0.9,
            )

        # Draw 2theta reference lines
        all_tths = [s["tth"] for s in spots]
        tth_min = max(0, min(all_tths) - 5) if all_tths else 60
        tth_max = min(180, max(all_tths) + 5) if all_tths else 130
        all_chis = [s["chi"] for s in spots]
        chi_min = min(all_chis) - 5 if all_chis else -50
        chi_max = max(all_chis) + 5 if all_chis else 50

        for tth_ref in np.arange(round(tth_min / 10) * 10, tth_max + 1, 10):
            ax.axvline(tth_ref, color="#1a2a3a", lw=0.6, ls="--", alpha=0.7, zorder=1)
            ax.text(
                tth_ref,
                chi_max + 0.5,
                f"{tth_ref:.0f}°",
                color="#445566",
                fontsize=6,
                ha="center",
            )

        ax.set_ylim(chi_min, chi_max)
        ax.set_xlim(tth_min, tth_max)

    # ── helper: gnomonic projection ───────────────────────────────────────────
    def draw_gnomonic(ax, spots, title):
        _style_angular_ax(ax, title)
        ax.set_xlabel("gX  =  −sin2θ·sinχ / (1+cos2θ)", color="#7788aa", fontsize=7)
        ax.set_ylabel("gY  =   sin2θ·cosχ / (1+cos2θ)", color="#7788aa", fontsize=7)
        ax.set_aspect("equal")

        fund = [s for s in spots if not s.get("is_superlattice", False)]
        super_ = [s for s in spots if s.get("is_superlattice", False)]

        for subset, mk, ec in [(fund, "o", COL_FUND), (super_, "*", COL_SUPER)]:
            if not subset:
                continue
            gXs = [_gnomonic(s["tth"], s["chi"])[0] for s in subset]
            gYs = [_gnomonic(s["tth"], s["chi"])[1] for s in subset]
            Es = [s["E"] / 1e3 for s in subset]
            sz = [max(4, 80 * s["intensity"] ** 0.4) for s in subset]
            ax.scatter(
                gXs,
                gYs,
                s=sz,
                c=Es,
                cmap=cmap,
                norm=E_norm,
                alpha=0.80,
                edgecolors=ec,
                linewidths=0.35,
                marker=mk,
                zorder=3,
            )

        # Label strongest fundamental spots
        for s in sorted(fund, key=lambda x: -x["intensity"])[:8]:
            h, k, l = s["hkl"]
            gx, gy = _gnomonic(s["tth"], s["chi"])
            ax.annotate(
                f"({h}{k}{l})",
                xy=(gx, gy),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=5.5,
                color="#aaccff",
                alpha=0.9,
            )

        for s in sorted(super_, key=lambda x: -x["intensity"])[:4]:
            h, k, l = s["hkl"]
            gx, gy = _gnomonic(s["tth"], s["chi"])
            ax.annotate(
                f"({h}{k}{l})",
                xy=(gx, gy),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=5.5,
                color=COL_SUPER,
                alpha=0.9,
            )

        # Reference circles: 2theta = 60, 70, 80, 90, 100, 110, 120 deg
        theta_circ = np.linspace(0, 2 * np.pi, 360)
        for tth_ref in range(60, 131, 10):
            gXc, gYc = _gnomonic(np.full(360, tth_ref), np.degrees(theta_circ))
            # Only draw arc where spots exist
            valid = np.isfinite(gXc) & np.isfinite(gYc)
            if valid.any():
                col = "#ffffaa" if tth_ref == 90 else "#1a2a3a"
                lw = 0.9 if tth_ref == 90 else 0.5
                ax.plot(
                    gXc[valid],
                    gYc[valid],
                    color=col,
                    lw=lw,
                    ls="--",
                    alpha=0.7,
                    zorder=1,
                )
                # Label
                ax.text(
                    0,
                    _gnomonic(tth_ref, 0)[1] + 0.02,
                    f"{tth_ref}°",
                    color="#445566" if tth_ref != 90 else "#ffffaa",
                    fontsize=5.5,
                    ha="center",
                    va="bottom",
                )

        # Chi reference lines (radial lines at chi = 0, ±30, ±60, ±90 deg)
        for chi_ref in [0, 30, -30, 60, -60, 90, -90]:
            # Draw radial line from origin
            r_max = 3.0
            gx_r = r_max * (-np.sin(np.radians(chi_ref)))  # at 2theta=90 gX = -sin(chi)
            gy_r = r_max * np.cos(np.radians(chi_ref))
            ax.plot(
                [0, gx_r],
                [0, gy_r],
                color="#1a2a3a",
                lw=0.5,
                ls=":",
                alpha=0.6,
                zorder=1,
            )
            if abs(chi_ref) <= 90:
                ax.text(
                    gx_r * 0.9,
                    gy_r * 0.9,
                    f"χ={chi_ref}°",
                    color="#334455",
                    fontsize=5.5,
                    ha="center",
                    va="center",
                )

        # Origin crosshair (forward beam)
        ax.plot(0, 0, "+", color="#ffffaa", ms=8, mew=1.2, zorder=6)

        # Auto-scale with margin
        all_gx = [_gnomonic(s["tth"], s["chi"])[0] for s in spots]
        all_gy = [_gnomonic(s["tth"], s["chi"])[1] for s in spots]
        all_gx = [v for v in all_gx if np.isfinite(v)]
        all_gy = [v for v in all_gy if np.isfinite(v)]
        if all_gx and all_gy:
            margin = 0.3
            xmin, xmax = min(all_gx) - margin, max(all_gx) + margin
            ymin, ymax = min(all_gy) - margin, max(all_gy) + margin
            # Keep square
            r = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) + margin
            ax.set_xlim(-r, r)
            ax.set_ylim(-r, r)

    # ── Draw all four panels ──────────────────────────────────────────────────
    n_super = sum(1 for s in spots_b2 if s.get("is_superlattice", False))

    draw_tth_chi(
        ax_bcc_ang, spots_bcc, f"BCC  Im-3m   –   2θ vs χ   ({len(spots_bcc)} spots)"
    )
    draw_gnomonic(ax_bcc_gno, spots_bcc, f"BCC  Im-3m   –   Gnomonic projection")
    draw_b2_title = (
        f"B2   Pm-3m   –   2θ vs χ   "
        f"({len(spots_b2)} spots, {n_super} superlattice)"
    )
    draw_tth_chi(ax_b2_ang, spots_b2, draw_b2_title)
    draw_gnomonic(ax_b2_gno, spots_b2, f"B2   Pm-3m   –   Gnomonic projection")

    # ── Shared legend ─────────────────────────────────────────────────────────
    leg = [
        Line2D(
            [0],
            [0],
            marker="o",
            lw=0,
            mfc=COL_FUND,
            mec=COL_FUND,
            ms=7,
            label="Fundamental (BCC & B2)",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            lw=0,
            mfc=COL_SUPER,
            mec=COL_SUPER,
            ms=9,
            label="B2 superlattice  (h+k+l odd)",
        ),
        Line2D(
            [0],
            [0],
            color="#ffffaa",
            lw=1,
            ls="--",
            label="2θ = 90°  (gnomonic unit circle)",
        ),
    ]
    fig.legend(
        handles=leg,
        loc="upper center",
        ncol=3,
        fontsize=8,
        framealpha=0.25,
        facecolor=BG,
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.97),
    )

    # ── Colourbar ─────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=E_norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=ax_cb)
    cb.set_label("E  (keV)", color="#8899aa", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="#8899aa", labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8899aa")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(
        0.5,
        0.995,
        "Laue pattern  –  AlCoCrFeNi  –  angular coordinates  "
        "(LaueTools convention: beam ∥ y,  χ = arctan(−uf_x / uf_z))",
        ha="center",
        va="top",
        fontsize=11,
        color="white",
        fontweight="bold",
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Angular plot saved -> {out_path}")
    return out_path


def plot_all(
    spots_bcc,
    spots_b2,
    crystal_bcc,
    camera,
    U,
    E_MIN_eV=5_000,
    E_MAX_eV=27_000,
    KI_HAT=np.array([1.0, 0.0, 0.0]),
    SOURCE_TYPE="bending_magnet",
    E_CRIT_eV=20_000,
):

    # Central 2theta of detector (LaueTools: 90 - xbet)
    tc, _ = camera.pixel_to_2theta_chi(camera.xcen, camera.ycen)
    n_super = sum(1 for s in spots_b2 if s.get("is_superlattice", False))
    all_E = [s["E"] for s in spots_bcc + spots_b2]
    E_norm = mcolors.Normalize(vmin=E_MIN_eV / 1e3, vmax=E_MAX_eV / 1e3)
    cmap = "plasma"
    cmap_obj = plt.get_cmap(cmap)

    fig = plt.figure(figsize=(22, 13))
    fig.patch.set_facecolor(BG)

    gs = mgridspec.GridSpec(
        2,
        5,
        width_ratios=[1.4, 1.4, 1, 1, 0.28],
        height_ratios=[1, 1],
        hspace=0.38,
        wspace=0.28,
        left=0.03,
        right=0.97,
        top=0.93,
        bottom=0.06,
    )

    ax_img_bcc = fig.add_subplot(gs[0, 0])
    ax_img_b2 = fig.add_subplot(gs[0, 1])
    ax_spec = fig.add_subplot(gs[0, 2])
    ax_sf = fig.add_subplot(gs[0, 3])
    ax_scatter = fig.add_subplot(gs[1, 0])
    ax_tth = fig.add_subplot(gs[1, 1])
    ax_geo = fig.add_subplot(gs[1, 2])
    ax_int = fig.add_subplot(gs[1, 3])
    ax_info = fig.add_subplot(gs[:, 4])

    Nh, Nv = camera.Nh, camera.Nv

    # ── helper: draw one detector image ───────────────────────────────────────

    draw_det_image(
        ax_img_bcc, spots_bcc, f"BCC detector image  ({len(spots_bcc)} spots)"
    )
    draw_det_image(
        ax_img_b2,
        spots_b2,
        f"B2 detector image  ({len(spots_b2)} spots,  " f"{n_super} superlattice)",
    )

    # ── Spectrum panel ─────────────────────────────────────────────────────────
    _ax_style(ax_spec, f'Synchrotron spectrum  ({SOURCE_TYPE.replace("_"," ")})')
    E_plot = np.linspace(max(500, E_MIN_eV * 0.4), E_MAX_eV * 1.1, 800)

    if SOURCE_TYPE in ("bending_magnet", "wiggler"):
        S = np.array([synchrotron_spectrum(E) for E in E_plot])
        S /= S.max()
        ax_spec.fill_between(E_plot / 1e3, S, alpha=0.18, color="#88aaff")
        ax_spec.plot(
            E_plot / 1e3,
            S,
            color="#88aaff",
            lw=1.4,
            label=f"Ec = {E_CRIT_eV/1e3:.1f} keV",
        )
        ax_spec.axvline(
            0.83 * E_CRIT_eV / 1e3,
            color="#88aaff",
            ls="--",
            lw=0.7,
            alpha=0.5,
            label=f"Peak ~0.83 Ec",
        )
    else:
        S_tot = np.zeros(len(E_plot))
        for n in range(1, 2 * N_HARMONICS, 2):
            En = n * E_FUNDAMENTAL_eV
            sig = En * HARMONIC_WIDTH
            Sh = (1 / n) * np.exp(-0.5 * ((E_plot - En) / sig) ** 2)
            S_tot += Sh
            if n <= 9:
                ax_spec.fill_between(
                    E_plot / 1e3, Sh / Sh.max() * 0.7, alpha=0.12, color="#88aaff"
                )
        if S_tot.max() > 0:
            S_tot /= S_tot.max()
        ax_spec.plot(E_plot / 1e3, S_tot, color="#88aaff", lw=1.4)

    # energy window
    ax_spec.axvspan(E_MIN_eV / 1e3, E_MAX_eV / 1e3, alpha=0.07, color="white")
    ax_spec.axvline(E_MIN_eV / 1e3, color="#888888", lw=0.7, ls="--")
    ax_spec.axvline(E_MAX_eV / 1e3, color="#888888", lw=0.7, ls="--")

    # spot energies as stems
    if spots_bcc:
        sw_arr = np.array([s["sw"] for s in spots_bcc])
        sw_norm = sw_arr / sw_arr.max() if sw_arr.max() > 0 else sw_arr
        ax_spec.vlines(
            [s["E"] / 1e3 for s in spots_bcc],
            0,
            sw_norm,
            color=COL_BCC,
            lw=0.4,
            alpha=0.35,
        )

    ax_spec.set_xlim(max(0.5, E_MIN_eV * 0.4 / 1e3), E_MAX_eV * 1.1 / 1e3)
    ax_spec.set_ylim(0, 1.3)
    ax_spec.set_xlabel("E  (keV)", color="#7788aa", fontsize=7)
    ax_spec.set_ylabel("S(E)  (norm.)", color="#7788aa", fontsize=7)
    ax_spec.legend(fontsize=6, framealpha=0.2, facecolor=BG, labelcolor="white")

    # ── |F(E)| panel ──────────────────────────────────────────────────────────
    _ax_style(ax_sf, "|F(G, E)|  vs energy  (BCC, top 4 spots)")
    E_arr = np.linspace(E_MIN_eV, E_MAX_eV, 500)
    plotted_E = []
    for s in sorted(spots_bcc, key=lambda x: -x["intensity"])[:6]:
        if any(abs(s["E"] - pe) < 1000 for pe in plotted_E):
            continue
        G = crystal_bcc.Q(*s["hkl"])
        FE = crystal_bcc.StructureFactorForEnergy(G, E_arr)
        col = cmap_obj(E_norm(s["E"] / 1e3))
        h, k, l = s["hkl"]
        ax_sf.plot(E_arr / 1e3, np.abs(FE), color=col, lw=1.1, label=f"({h}{k}{l})")
        ax_sf.axvline(s["E"] / 1e3, color=col, lw=0.6, ls="--", alpha=0.4)
        plotted_E.append(s["E"])
    ax_sf.set_xlabel("E  (keV)", color="#7788aa", fontsize=7)
    ax_sf.set_ylabel("|F|  (e.u.)", color="#7788aa", fontsize=7)
    ax_sf.legend(fontsize=6, framealpha=0.2, facecolor=BG, labelcolor="white")

    # ── Scatter plot col/row coloured by energy ────────────────────────────────
    _ax_style(ax_scatter, "Spot map  (pixel coordinates, coloured by E)")
    ax_scatter.set_facecolor("#04060e")
    ax_scatter.set_xlim(0, Nh)
    ax_scatter.set_ylim(Nv, 0)
    ax_scatter.set_aspect("equal")
    ax_scatter.set_xlabel("col  (pixel)", color="#7788aa", fontsize=7)
    ax_scatter.set_ylabel("row  (pixel)", color="#7788aa", fontsize=7)

    # Draw detector outline
    ax_scatter.add_patch(
        Rectangle((0, 0), Nh, Nv, fill=False, edgecolor="#334466", lw=0.8)
    )

    for spots_s, mk, ec in [
        (spots_bcc, "o", COL_BCC),
        ([s for s in spots_b2 if s.get("is_superlattice", False)], "*", COL_SUP),
    ]:
        if not spots_s:
            continue
        cs = [s["pix"][0] for s in spots_s]
        rs = [s["pix"][1] for s in spots_s]
        Es = [s["E"] / 1e3 for s in spots_s]
        sz = [max(3, 60 * s["intensity"] ** 0.4) for s in spots_s]
        ax_scatter.scatter(
            cs,
            rs,
            s=sz,
            c=Es,
            cmap=cmap,
            norm=E_norm,
            alpha=0.75,
            edgecolors=ec,
            linewidths=0.3,
            marker=mk,
            zorder=3,
        )

    # Centre and direct beam
    ax_scatter.plot(Nh / 2, Nv / 2, "+", color="#aaaaff", ms=8, mew=1, zorder=6)
    ki_hat = KI_HAT / np.linalg.norm(KI_HAT)
    db = camera.project(ki_hat)
    if db:
        ax_scatter.plot(*db, "x", color=COL_DB, ms=10, mew=1.5, zorder=7)

    # 2theta grid lines
    CC, RR, TTH_g = camera.tth_grid(step=max(1, Nh // 20))
    tc, _ = camera.pixel_to_2theta_chi(camera.xcen, camera.ycen)
    lvls = sorted({tc - 20, tc - 10, tc, tc + 10, tc + 20})
    lvls = [l for l in lvls if TTH_g.min() < l < TTH_g.max()]
    if lvls:
        ct = ax_scatter.contour(
            CC, RR, TTH_g, levels=lvls, colors="#1a2a44", linewidths=0.6, alpha=0.8
        )
        ax_scatter.clabel(ct, fmt="%.0f°", fontsize=5, colors="#3355aa")

    leg_sc = [
        Line2D(
            [0],
            [0],
            marker="o",
            lw=0,
            mfc=COL_BCC,
            mec=COL_BCC,
            ms=5,
            label="BCC fundamental",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            lw=0,
            mfc=COL_SUP,
            mec=COL_SUP,
            ms=7,
            label="B2 superlattice",
        ),
        Line2D(
            [0],
            [0],
            marker="+",
            lw=0,
            color="#aaaaff",
            ms=6,
            mew=1,
            label="Det. centre",
        ),
        Line2D(
            [0], [0], marker="x", lw=0, color=COL_DB, ms=6, mew=1.3, label="Direct beam"
        ),
    ]
    ax_scatter.legend(
        handles=leg_sc,
        fontsize=5.5,
        framealpha=0.2,
        facecolor=BG,
        labelcolor="white",
        loc="upper right",
    )

    # ── 2theta histogram ──────────────────────────────────────────────────────
    _ax_style(ax_tth, "2theta distribution (intensity-weighted)")
    bins = np.linspace(0, 180, 72)
    if spots_bcc:
        ax_tth.hist(
            [s["tth"] for s in spots_bcc],
            bins=bins,
            weights=[s["intensity"] for s in spots_bcc],
            color=COL_BCC,
            alpha=0.55,
            label="BCC fund.",
        )
    sup = [s for s in spots_b2 if s.get("is_superlattice", False)]
    if sup:
        ax_tth.hist(
            [s["tth"] for s in sup],
            bins=bins,
            weights=[s["intensity"] for s in sup],
            color=COL_SUP,
            alpha=0.70,
            label="B2 superlat.",
        )
    ax_tth.axvline(tc, color="#ffffaa", lw=1, ls="--", label=f"Det. centre = {tc:.0f}°")
    ax_tth.set_xlabel("2theta  (deg)", color="#7788aa", fontsize=7)
    ax_tth.set_ylabel("Sum intensity", color="#7788aa", fontsize=7)
    ax_tth.legend(fontsize=6, framealpha=0.2, facecolor=BG, labelcolor="white")

    # ── Geometry schematic ────────────────────────────────────────────────────
    _ax_style(ax_geo, "Geometry  (top view: x-y plane)")
    ax_geo.set_facecolor("#04060e")
    ax_geo.set_xlim(-1.7, 2.5)
    ax_geo.set_ylim(-1.8, 1.8)
    ax_geo.set_aspect("equal")
    ax_geo.axis("off")

    # beam – draw from negative KI_HAT direction
    ki_2d = np.array([KI_HAT[0], KI_HAT[1]])
    if np.linalg.norm(ki_2d) > 1e-6:
        ki_2d /= np.linalg.norm(ki_2d)
    else:
        ki_2d = np.array([0.0, 1.0])
    ax_geo.annotate(
        "",
        xy=(0, 0),
        xytext=tuple(-1.6 * ki_2d),
        arrowprops=dict(arrowstyle="->", color=COL_DB, lw=2.2),
    )
    ki_str = "".join([f"{v:+.2g}" if v != 0 else "" for v in KI_HAT])
    ax_geo.text(
        -0.8, 0.14, f"white beam ({ki_str})", color=COL_DB, fontsize=7.5, ha="center"
    )

    # sample
    ax_geo.add_patch(
        plt.Polygon(
            [(-0.12, -0.22), (0.12, -0.22), (0.12, 0.22), (-0.12, 0.22)],
            color="#445599",
            zorder=3,
        )
    )
    ax_geo.text(0, -0.38, "crystal", color="#aabbdd", fontsize=7, ha="center")

    # detector: draw as a rotated rectangle representing its orientation
    tth_c = np.radians(tc)
    # In 2D schematic (x-y plane): rotate KI by tth_c to get detector direction
    ki_2d_n = ki_2d  # already defined above (normalised 2D projection of KI_HAT)
    c_tth, s_tth = np.cos(tth_c), np.sin(tth_c)
    det_dir = np.array(
        [
            c_tth * ki_2d_n[0] - s_tth * ki_2d_n[1],
            s_tth * ki_2d_n[0] + c_tth * ki_2d_n[1],
        ]
    )
    det_perp = np.array([det_dir[1], -det_dir[0]])

    L = 1.4  # diagram scale
    half_det = 0.45
    dc_diag = L * det_dir
    p1 = dc_diag + half_det * det_perp
    p2 = dc_diag - half_det * det_perp
    ax_geo.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        color="#888899",
        lw=6,
        solid_capstyle="round",
        alpha=0.75,
    )
    ax_geo.text(
        dc_diag[0] + det_dir[0] * 0.22,
        dc_diag[1] + det_dir[1] * 0.22,
        "detector",
        color="#888899",
        fontsize=6.5,
        ha="center",
        va="center",
    )

    # scattered beams at a few angles
    for tth_s, col_s, lbl in [
        (tc, "#ffffaa", f"2th={tc:.0f}deg (centre)"),
        (tc - 15, "#88ddaa", f"2th={tc-15:.0f}deg"),
        (tc + 15, "#ffaa66", f"2th={tc+15:.0f}deg"),
    ]:
        if 5 < tth_s < 175:
            tr = np.radians(tth_s)
            c_s, s_s = np.cos(tr), np.sin(tr)
            # Rotate KI_HAT by tth_s
            kf_2d = np.array(
                [
                    c_s * ki_2d_n[0] - s_s * ki_2d_n[1],
                    s_s * ki_2d_n[0] + c_s * ki_2d_n[1],
                ]
            )
            ax_geo.annotate(
                "",
                xy=(kf_2d[0] * L * 0.85, kf_2d[1] * L * 0.85),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=col_s, lw=1.2),
            )
            ax_geo.text(
                kf_2d[0] * L * 0.9 + 0.05,
                kf_2d[1] * L * 0.9,
                lbl,
                color=col_s,
                fontsize=5.5,
                ha="left",
                va="center",
            )

    # 2theta arc (around KI direction)
    ki_angle = np.arctan2(ki_2d_n[1], ki_2d_n[0])  # angle of KI in 2D
    arc_angles = np.linspace(
        ki_angle + np.radians(max(5, tc - 25)),
        ki_angle + np.radians(min(175, tc + 25)),
        80,
    )
    ax_geo.plot(
        0.65 * np.cos(arc_angles),
        0.65 * np.sin(arc_angles),
        color="#334455",
        lw=1,
        ls="--",
    )
    mid_arc = ki_angle + np.radians(tc)
    ax_geo.text(
        0.75 * np.cos(mid_arc),
        0.75 * np.sin(mid_arc),
        "2θ",
        color="#556677",
        fontsize=9,
    )

    # beam direction label
    bd = beam_in_crystal(U)
    ax_geo.text(
        0,
        -1.7,
        f"beam \u2225 [{bd[0]:.2g},{bd[1]:.2g},{bd[2]:.2g}]  "
        f"\u03c6\u2081={PHI1_DEG:.0f}\u00b0 \u03a6={PHI_DEG:.0f}\u00b0 "
        f"\u03c6\u2082={PHI2_DEG:.0f}\u00b0",
        color="#aaaacc",
        fontsize=6.5,
        ha="center",
    )

    # ── Intensity vs pixel column (horizontal cross-section) ──────────────────
    _ax_style(ax_int, "Intensity vs. 2theta (all spots)")
    if spots_bcc:
        tths_b = [s["tth"] for s in spots_bcc]
        intn_b = [s["intensity"] for s in spots_bcc]
        Es_b = [s["E"] / 1e3 for s in spots_bcc]
        ax_int.scatter(
            tths_b, intn_b, s=6, c=Es_b, cmap=cmap, norm=E_norm, alpha=0.6, zorder=3
        )
    if sup:
        ax_int.scatter(
            [s["tth"] for s in sup],
            [s["intensity"] for s in sup],
            s=15,
            color=COL_SUP,
            marker="*",
            alpha=0.85,
            zorder=4,
            label="B2 superlat.",
        )
    ax_int.axvline(tc, color="#ffffaa", lw=0.8, ls="--", label=f"Centre {tc:.0f}°")
    ax_int.set_xlabel("2theta  (deg)", color="#7788aa", fontsize=7)
    ax_int.set_ylabel("I / I_max", color="#7788aa", fontsize=7)
    ax_int.legend(fontsize=6, framealpha=0.2, facecolor=BG, labelcolor="white")

    # ── Colour bar ────────────────────────────────────────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=E_norm)
    sm.set_array([])
    cb_ax = fig.add_axes([0.805, 0.56, 0.008, 0.34])
    cb = fig.colorbar(sm, cax=cb_ax)
    cb.set_label("E  (keV)", color="#8899aa", fontsize=7)
    cb.ax.yaxis.set_tick_params(color="#8899aa", labelsize=6)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8899aa")

    # ── Info panel ────────────────────────────────────────────────────────────
    ax_info.set_facecolor("#0b0f1c")
    ax_info.axis("off")
    bd = beam_in_crystal(U)

    src_detail = (
        f"Ec={E_CRIT_eV/1e3:.1f} keV"
        if SOURCE_TYPE in ("bending_magnet", "wiggler")
        else f"E1={E_FUNDAMENTAL_eV/1e3:.1f} keV"
    )

    lines = [
        ("AlCoCrFeNi  HEA", 12, "white", True),
        ("White-Beam Laue Reflection", 9, "#aaaaff", True),
        ("", 0, "", False),
        ("Source ──────────────────", 7, "#334466", False),
        (SOURCE_TYPE.replace("_", " "), 7, "#88aaff", False),
        (src_detail, 7, "#88aaff", False),
        (f"{E_MIN_eV/1e3:.0f}-{E_MAX_eV/1e3:.0f} keV window", 7, "#88aaff", False),
        ("", 0, "", False),
        ("Crystal ─────────────────", 7, "#334466", False),
        (
            f"phi1={PHI1_DEG:.1f} Phi={PHI_DEG:.1f} phi2={PHI2_DEG:.1f} deg",
            7,
            "#88aaff",
            False,
        ),
        (f"beam||[{bd[0]:.2g},{bd[1]:.2g},{bd[2]:.2g}]", 7, "#88aaff", False),
        (
            f"ki=[{KI_HAT[0]:.2g},{KI_HAT[1]:.2g},{KI_HAT[2]:.2g}] lab",
            7,
            "#88aaff",
            False,
        ),
        (f"a = {A_LATTICE} Ang", 7, "#88aaff", False),
        ("", 0, "", False),
        ("Camera ──────────────────", 7, "#334466", False),
        (f"{camera.Nh} x {camera.Nv} pixels", 7, "#88aaff", False),
        (f"pixel = {camera.pixel_mm*1e3:.0f} um", 7, "#88aaff", False),
        (
            f"size {camera.size_h_mm:.0f} x {camera.size_v_mm:.0f} mm^2",
            7,
            "#88aaff",
            False,
        ),
        (f"dist = {camera.dd:.1f} mm", 7, "#88aaff", False),
        (f"2th at (xcen,ycen) = {90-camera.xbet:.2f} deg", 7, "#88aaff", False),
        (f"xbet = {camera.xbet:.3f} deg", 7, "#88aaff", False),
        (f"xgam = {camera.xgam:.3f} deg", 7, "#88aaff", False),
        ("", 0, "", False),
        ("Results ─────────────────", 7, "#334466", False),
        (f"BCC : {len(spots_bcc)} spots", 8, "#4fc3f7", False),
        (f"B2  : {len(spots_b2)} spots", 8, "#ffb74d", False),
        (f"  fund. : {len(spots_b2)-n_super}", 7, "#88aaff", False),
        (f"  superl: {n_super}", 7, "#ff6633", False),
        ("", 0, "", False),
        ("Intensity ────────────────", 7, "#334466", False),
        ("I=|F(Q,E)|^2*LP*S(E)", 7, "#88aaff", False),
        ("Cromer-Mann+Henke f',f\"", 6, "#556677", False),
        ("LP=(1+cos^2 2T)/(2s^2 c)", 6, "#556677", False),
        ("Gaussian spot profile", 6, "#556677", False),
    ]

    y = 0.98
    for txt, fs, col, bold in lines:
        if txt == "":
            y -= 0.010
            continue
        ax_info.text(
            0.04,
            y,
            txt,
            transform=ax_info.transAxes,
            fontsize=fs,
            color=col,
            fontweight="bold" if bold else "normal",
            va="top",
            fontfamily="monospace",
        )
        y -= 0.030 if fs >= 9 else 0.024 if fs >= 7 else 0.020

    # ── Title ─────────────────────────────────────────────────────────────────
    bd_str = f"[{bd[0]:.2g},{bd[1]:.2g},{bd[2]:.2g}]"
    src_str = SOURCE_TYPE.replace("_", " ")
    fig.text(
        0.5,
        0.965,
        f"White-Beam Laue  |  AlCoCrFeNi  |  {src_str}  "
        f"{E_MIN_eV/1e3:.0f}-{E_MAX_eV/1e3:.0f} keV  |  "
        f"beam || {bd_str}  |  "
        f"2theta_centre = {tc:.0f}deg  |  "
        f"{camera.Nh}x{camera.Nv} px  {camera.pixel_mm*1e3:.0f}um",
        ha="center",
        fontsize=11,
        color="white",
        fontweight="bold",
    )

    IMAGE_OUTPUT = "laue_white_synchrotron.png"
    plt.savefig(
        IMAGE_OUTPUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    print(f"\n  Figure saved -> {IMAGE_OUTPUT}")


def plot_layer_contributions(
    spots,
    stack,
    camera,
    space="detector",
    image=None,
    out_path=None,
):
    """
    Visualise per-layer intensity contributions, either across the detector
    image or in 2theta/chi space.

    Produces a figure with one scatter panel per layer, coloured by each
    layer's fractional intensity contribution at that spot position, plus a
    summary panel showing the dominant-layer map and a bar chart of mean
    per-layer contribution. All position panels (per-layer + dominant-layer)
    share their x/y axes, and the panel grid is laid out as close to square
    as possible for the number of layers.

    Args:
        spots (list of dicts from `layer_contributions_spots()`):
        stack (LayeredCrystal):
        camera (Camera):
        space : {'detector', 'angular'}, optional
            Coordinate system for the position panels. `'detector'`
            (default) plots detector pixel position (col vs row).
            `'angular'` plots 2theta vs chi (2θ on x, χ on y -- LaueTools
            `.cor` file convention).
        image (ndarray of shape (Nv, Nh) or None): Optional raw/rendered detector image (e.g. from `camera.render()`
            or a real measured frame) shown as a dimmed background on every
            position panel, log-normalised with the `'inferno'` colormap --
            same convention as :func:`plot_measured_vs_simulated`. Only
            used when `space='detector'`; ignored (with a note printed)
            for `space='angular'`.
        out_path (str, optional): Save figure to this path if provided.  If
            `None` (default), the figure is not saved to disk.

    Returns:
        fig
"""
    import matplotlib.colors as mcolors
    import matplotlib.gridspec as mgridspec
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    if space not in ("detector", "angular"):
        raise ValueError(f"space must be 'detector' or 'angular', got {space!r}")

    img_arr, img_norm = None, None
    if image is not None:
        if space != "detector":
            print("  Note: `image` is only drawn for space='detector'; ignored for 'angular'.")
        else:
            img_arr = np.asarray(image, dtype=float)
            if img_arr.shape != (camera.Nv, camera.Nh):
                raise ValueError(
                    f"image shape {img_arr.shape} does not match "
                    f"(camera.Nv, camera.Nh) = {(camera.Nv, camera.Nh)}"
                )
            vmin = max(img_arr[img_arr > 0].min(), 1.0) if np.any(img_arr > 0) else 1.0
            img_norm = mcolors.LogNorm(vmin=vmin, vmax=img_arr.max())

    if not spots or "layer_I_frac" not in spots[0]:
        raise ValueError("Call layer_contributions_spots(spots, stack) first.")

    # De-duplicated, first-seen order: a label can legitimately appear on
    # more than one Layer object (e.g. the same material reused across two
    # different repeating blocks), but layer_contributions_spots() already
    # merges those into a single layer_I_frac[label] entry -- one panel per
    # label, not per Layer instance.
    labels = list(dict.fromkeys(layer.label for layer in stack.layers))
    n_layers = len(labels)

    # Layer colours
    layer_cols = [
        "#4fc3f7",
        "#ff9f43",
        "#ff6b6b",
        "#a29bfe",
        "#55efc4",
        "#fd79a8",
        "#fdcb6e",
    ][:n_layers]

    BG = "#080c14"

    # ── Position-panel coordinate system ────────────────────────────────────
    if space == "detector":
        Nh, Nv = camera.Nh, camera.Nv
        xlim, ylim = (0, Nh), (Nv, 0)
        xlabel, ylabel = "col", "row"

        def get_xy(s):
            return s["pix"][0], s["pix"][1]

    else:
        tths_all = [s["tth"] for s in spots]
        chis_all = [s["chi"] for s in spots]
        xlim = (min(tths_all) - 3, max(tths_all) + 3) if tths_all else (60, 130)
        ylim = (min(chis_all) - 3, max(chis_all) + 3) if chis_all else (-50, 50)
        xlabel, ylabel = "2theta (deg)", "chi (deg)"

        def get_xy(s):
            return s["tth"], s["chi"]

    def style_position_ax(ax, title, title_color):
        ax.set_facecolor("#04060e")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(title, color=title_color, fontsize=8, pad=4)
        ax.set_xlabel(xlabel, color="#7788aa", fontsize=6)
        ax.set_ylabel(ylabel, color="#7788aa", fontsize=6)
        ax.tick_params(colors="#7788aa", labelsize=5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a1f2e")
        if space == "angular":
            ax.grid(True, ls=":", lw=0.3, color="#181e2e")
            ax.axhline(0, color="#222244", lw=0.5)
        if img_arr is not None:
            ax.imshow(
                img_arr,
                origin="upper",
                cmap="inferno",
                norm=img_norm,
                aspect="auto",
                interpolation="antialiased",
                filternorm=True,
                alpha=0.6,
                zorder=1,
            )

    # ── Squarish grid layout: n_layers position panels + dominant-layer
    # summary + bar chart ────────────────────────────────────────────────────
    n_slots = n_layers + 2
    ncols = int(np.ceil(np.sqrt(n_slots)))
    nrows = int(np.ceil(n_slots / ncols))

    fig = plt.figure(figsize=(4.6 * ncols, 4.2 * nrows))
    fig.patch.set_facecolor(BG)

    gs = mgridspec.GridSpec(
        nrows,
        ncols,
        hspace=0.4,
        wspace=0.3,
        left=0.04,
        right=0.97,
        top=0.90,
        bottom=0.06,
    )

    def slot(i, **kw):
        r, c = divmod(i, ncols)
        return fig.add_subplot(gs[r, c], **kw)

    # ── Per-layer position panels (sharex/sharey linked) ────────────────────
    shared_ax = None
    for li, (label, col) in enumerate(zip(labels, layer_cols)):
        ax = slot(li, sharex=shared_ax, sharey=shared_ax)
        shared_ax = shared_ax or ax
        style_position_ax(ax, label, col)

        sel = [s for s in spots if label in s.get("layer_I_frac", {})]
        if sel:
            xs, ys = zip(*(get_xy(s) for s in sel))
            cs = [max(-1, min(1, s["layer_I_frac"][label])) for s in sel]
            sz = [max(2, 40 * s["intensity"] ** 0.4) for s in sel]
            sc = ax.scatter(
                xs,
                ys,
                s=sz,
                c=cs,
                cmap="RdYlGn",
                vmin=-0.2,
                vmax=1.0,
                alpha=0.85,
                edgecolors="none",
                zorder=3,
            )
            cbar = plt.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
            cbar.set_label("I_frac", color="#7788aa", fontsize=5)
            cbar.ax.tick_params(colors="#7788aa", labelsize=5)

        if space == "detector":
            CC, RR, TTH = camera.tth_grid(step=max(1, Nh // 15))
            tc, _ = camera.pixel_to_2theta_chi(camera.xcen, camera.ycen)
            lvls = [tc - 20, tc, tc + 20]
            lvls = [l for l in lvls if TTH.min() < l < TTH.max()]
            if lvls:
                ax.contour(
                    CC, RR, TTH, levels=lvls, colors="#1a2a3a", linewidths=0.5, alpha=0.6
                )

    # ── Dominant-layer summary panel (same shared axes) ─────────────────────
    ax_dom = slot(n_layers, sharex=shared_ax, sharey=shared_ax)
    style_position_ax(ax_dom, "Dominant layer", "#ccccee")

    for s in spots:
        if "layer_I_frac" not in s:
            continue
        dom_idx = max(
            range(n_layers), key=lambda i: s["layer_I_frac"].get(labels[i], -np.inf)
        )
        col = layer_cols[dom_idx]
        sz = max(2, 50 * s["intensity"] ** 0.4)
        x, y = get_xy(s)
        ax_dom.scatter(x, y, s=sz, color=col, alpha=0.8, edgecolors="none", zorder=3)

    leg = [
        Line2D([0], [0], marker="o", lw=0, mfc=layer_cols[i], ms=6, label=labels[i])
        for i in range(n_layers)
    ]
    ax_dom.legend(
        handles=leg,
        fontsize=5.5,
        framealpha=0.25,
        facecolor=BG,
        labelcolor="white",
        loc="upper right",
    )

    # ── Bar chart: mean intensity fraction per layer (own axes) ─────────────
    ax_bar = slot(n_layers + 1)
    ax_bar.set_facecolor("#04060e")
    for sp in ax_bar.spines.values():
        sp.set_edgecolor("#1a1f2e")
    ax_bar.tick_params(colors="#7788aa", labelsize=7)

    avg_fracs = {}
    for label in labels:
        fracs = [s["layer_I_frac"].get(label, 0) for s in spots if "layer_I_frac" in s]
        avg_fracs[label] = np.mean(fracs) * 100 if fracs else 0.0

    ax_bar.barh(labels, [avg_fracs[l] for l in labels], color=layer_cols, alpha=0.8)
    ax_bar.axvline(0, color="#555566", lw=0.7)
    ax_bar.set_xlabel("Mean intensity fraction (%)", color="#7788aa", fontsize=7)
    ax_bar.set_title("Mean layer contribution", color="#ccccee", fontsize=8, pad=4)

    # ── Hide unused trailing grid slots ──────────────────────────────────────
    for i in range(n_slots, nrows * ncols):
        slot(i).axis("off")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(
        0.5,
        0.96,
        f"Per-layer intensity decomposition ({space})  |  {stack.name}  |  "
        f"{len(spots)} spots  |  total thickness {stack.total_thickness / 1e4:.3f} µm",
        ha="center",
        fontsize=10,
        color="white",
        fontweight="bold",
    )

    if out_path is not None:
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Figure -> {out_path}")
    return fig


def draw_det_image(
    ax,
    spots,
    camera,
    title,
    KI_HAT=np.array([1.0, 0.0, 0.0]),
    sigma=2.0,
    normalize=False,
):
    img = camera.render(spots, sigma_pix=sigma, log_scale=True, normalize=normalize)

    Nh, Nv = camera.Nh, camera.Nv
    im = ax.imshow(
        img.T,
        origin="upper",
        cmap="hot",
        extent=[0, Nv, Nh, 0],
        # aspect="auto",
        interpolation="nearest",
    )

    # Axis labels in mm and pixels
    ax.set_xlabel("row  (pixel)", color="#7788aa", fontsize=7)
    ax.set_ylabel(
        f"col  (pixel,  pitch={camera.pixel_mm*1e3:.0f} µm)",
        color="#7788aa",
        fontsize=7,
    )

    # 2theta contours overlaid on detector image
    # Sample a grid of pixels, compute 2theta, contour
    CC, RR, TTH = camera.tth_grid(step=max(1, Nh // 20))

    # Contour levels around the centre 2theta
    tc, _ = camera.pixel_to_2theta_chi(camera.xcen, camera.ycen)
    levels = [tc - 20, tc - 10, tc, tc + 10, tc + 20]
    levels = [l for l in levels if TTH.min() < l < TTH.max()]
    if levels:
        ct = ax.contour(
            RR, CC, TTH, levels=levels, colors="#2244aa", linewidths=0.5, alpha=0.6
        )
        ax.clabel(ct, fmt="%.0f°", fontsize=5, colors="#4466cc")

    # Direct beam marker (if on detector)
    ki_hat = KI_HAT / np.linalg.norm(KI_HAT)
    db = camera.project(ki_hat)
    if db:
        ax.plot(db[1], db[0], "x", color=COL_DB, ms=8, mew=1.3, zorder=8)

    # Centre cross
    ax.plot(Nv / 2, Nh / 2, "+", color="#aaaaff", ms=6, mew=0.8, zorder=7)

    _ax_style(ax, title)
    ax.set_xlim(0, Nv)
    ax.set_ylim(Nh, 0)

    # Tick labels in mm
    def mm_fmt_h(x, pos):
        return f"{(x-Nv/2)*camera.pixel_mm:.0f}"

    def mm_fmt_v(x, pos):
        return f"{(x-Nh/2)*camera.pixel_mm:.0f}"

    ax2h = ax.secondary_xaxis(
        "top",
        functions=(
            lambda r: (r - Nv / 2) * camera.pixel_mm,
            lambda m: m / camera.pixel_mm + Nv / 2,
        ),
    )
    ax2v = ax.secondary_yaxis(
        "right",
        functions=(
            lambda c: (c - Nh / 2) * camera.pixel_mm,
            lambda m: m / camera.pixel_mm + Nh / 2,
        ),
    )
    ax2h.set_xlabel("mm from centre", color="#7788aa", fontsize=6)
    ax2v.set_ylabel("mm from centre", color="#7788aa", fontsize=6)
    ax2h.tick_params(colors="#7788aa", labelsize=5)
    ax2v.tick_params(colors="#7788aa", labelsize=5)
    for sp in ax2h.spines.values():
        sp.set_edgecolor("#1a1f2e")
    for sp in ax2v.spines.values():
        sp.set_edgecolor("#1a1f2e")

    # Colorbar
    cb = ax.get_figure().colorbar(im, ax=ax, location="left", pad=0.12, fraction=0.03)
    cb.set_label(
        "log intensity" if not normalize else "normalised log intensity",
        color="#7788aa",
        fontsize=6,
    )
    cb.ax.tick_params(colors="#7788aa", labelsize=5)
    cb.outline.set_edgecolor("#1a1f2e")


# ─────────────────────────────────────────────────────────────────────────────
# STRAIN BROADENING PLOT
# ─────────────────────────────────────────────────────────────────────────────

_VOIGT_LABELS = ["ε₁₁", "ε₂₂", "ε₃₃", "ε₂₃", "ε₁₃", "ε₁₂"]


def plot_strain_broadening(
    spots_b,
    camera,
    jacobians=None,
    out_path="strain_broadening.png",
    top_n=12,
):
    """
    Three-panel summary of strain-induced spot broadening.

    Panel A — Detector map
        Each spot is drawn as an ellipse whose semi-axes come from the
        eigenvalues of the 2×2 pixel-space covariance `cov_pix`, coloured
        by `sigma_strain_pix` (semi-major axis).

    Panel B — σ_strain vs 2θ
        Scatter plot of the major (solid) and minor (dashed) broadening
        semi-axes versus 2θ for every spot.  The most-broadened spots are
        labelled with their (hkl) index.

    Panel C — Jacobian heat-map  *(only when* `jacobians` *is provided)*
        Rows = top_n most-broadened spots; columns = the 6 Voigt strain
        components.  Cell colour = |∂xcam/∂εᵢⱼ| or |∂ycam/∂εᵢⱼ|
        (RMS of both rows of J), so you can read off which strain components
        most affect which spots.

    Args:
        spots_b (list of dict): Output of :func:`~nrxrdct.laue.simulation.strain_broadening`.
            Must contain `'cov_pix'`, `'sigma_strain_pix'`,
            `'sigma_strain_minor'`, `'pix'`, `'tth'`, `'hkl'`.
        camera (Camera): Used for detector dimensions in Panel A.
        jacobians (dict {(h,k,l): ndarray (2,6)}, optional): Output of :func:`~nrxrdct.laue.simulation.strain_spot_jacobian`.
            When supplied, Panel C is drawn; otherwise it is replaced with a
            colour-bar for Panel A.
        out_path (str, optional): File path to save the figure.  `None` → do not save.
        top_n (int, optional): Number of most-broadened spots to label / show in Panel C.

    Returns:
        fig (matplotlib.figure.Figure):
"""
    from matplotlib.patches import Ellipse

    has_jac = jacobians is not None
    ncols = 3 if has_jac else 2
    fig, axes = plt.subplots(
        1, ncols,
        figsize=(6 * ncols, 6),
        gridspec_kw={"width_ratios": [2, 1.2, 1.4] if has_jac else [2, 1.2]},
    )
    fig.patch.set_facecolor(BG)
    ax_det, ax_scatter = axes[0], axes[1]
    ax_jac = axes[2] if has_jac else None

    # ── colour scale (shared) ─────────────────────────────────────────────────
    sigmas = [s["sigma_strain_pix"] for s in spots_b if s.get("pix") is not None]
    vmax = max(sigmas) if sigmas else 1.0
    cmap_s = "inferno"
    norm_s = mcolors.Normalize(vmin=0, vmax=vmax)

    # ── Panel A: detector map ─────────────────────────────────────────────────
    ax_det.set_facecolor(BG)
    ax_det.set_xlim(0, camera.Nh)
    ax_det.set_ylim(camera.Nv, 0)          # pixel row increases downward
    ax_det.set_aspect("equal")
    ax_det.set_xlabel("xcam  (pixel)", color=FG, fontsize=8)
    ax_det.set_ylabel("ycam  (pixel)", color=FG, fontsize=8)
    ax_det.tick_params(colors="#7788aa", labelsize=6)
    ax_det.set_title("Detector: strain broadening ellipses", color=FG, fontsize=9)
    for sp in ax_det.spines.values():
        sp.set_edgecolor("#1a1f2e")

    for s in spots_b:
        if s.get("pix") is None:
            continue
        xc, yc = s["pix"]
        cov = s.get("cov_pix", np.zeros((2, 2)))
        sigma_maj = s.get("sigma_strain_pix", 0.0)
        col = plt.cm.get_cmap(cmap_s)(norm_s(sigma_maj))

        if sigma_maj < 0.05:
            # Too small to show as ellipse — draw a dot
            ax_det.plot(xc, yc, ".", ms=2, color=col, alpha=0.7)
            continue

        # Compute ellipse orientation from eigenvectors of cov_pix
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        angle_deg = np.degrees(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))

        ell = Ellipse(
            xy=(xc, yc),
            width=2 * np.sqrt(eigvals[-1]),   # 1-sigma semi-major (diameter)
            height=2 * np.sqrt(eigvals[0]),   # 1-sigma semi-minor
            angle=angle_deg,
            linewidth=0.8,
            edgecolor=col,
            facecolor="none",
            alpha=0.85,
            zorder=3,
        )
        ax_det.add_patch(ell)
        ax_det.plot(xc, yc, "+", ms=3, color=col, lw=0.5, alpha=0.6)

    # Label top_n broadened spots
    top_spots = sorted(
        [s for s in spots_b if s.get("pix") is not None],
        key=lambda s: -s.get("sigma_strain_pix", 0),
    )[:top_n]
    for s in top_spots:
        h, k, l = s["hkl"]
        xc, yc = s["pix"]
        ax_det.annotate(
            f"({h}{k}{l})",
            xy=(xc, yc), xytext=(4, -4),
            textcoords="offset points",
            fontsize=5, color="#aaccff", alpha=0.9,
        )

    sm = plt.cm.ScalarMappable(cmap=cmap_s, norm=norm_s)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax_det, fraction=0.03, pad=0.02)
    cb.set_label("σ_strain  (px, semi-major)", color="#7788aa", fontsize=7)
    cb.ax.tick_params(colors="#7788aa", labelsize=6)
    cb.outline.set_edgecolor("#1a1f2e")

    # ── Panel B: σ_strain vs 2θ ───────────────────────────────────────────────
    ax_scatter.set_facecolor(BG)
    ax_scatter.tick_params(colors="#7788aa", labelsize=6)
    ax_scatter.set_xlabel("2θ  (degrees)", color=FG, fontsize=8)
    ax_scatter.set_ylabel("broadening  (px, 1σ)", color=FG, fontsize=8)
    ax_scatter.set_title("Broadening vs 2θ", color=FG, fontsize=9)
    for sp in ax_scatter.spines.values():
        sp.set_edgecolor("#1a1f2e")
    ax_scatter.grid(True, ls=":", lw=0.3, color="#181e2e")

    valid = [s for s in spots_b if s.get("pix") is not None]
    tths = [s["tth"] for s in valid]
    sig_maj = [s["sigma_strain_pix"] for s in valid]
    sig_min = [s["sigma_strain_minor"] for s in valid]
    cols = [plt.cm.get_cmap(cmap_s)(norm_s(v)) for v in sig_maj]

    ax_scatter.scatter(tths, sig_maj, c=cols, s=18, zorder=3, label="semi-major σ")
    ax_scatter.scatter(tths, sig_min, c=cols, s=8, marker="^", alpha=0.5,
                       zorder=2, label="semi-minor σ")

    # Connect major/minor for each spot
    for t, sj, sn in zip(tths, sig_maj, sig_min):
        ax_scatter.plot([t, t], [sn, sj], color="#334455", lw=0.4, zorder=1)

    # Label the top_n most broadened
    top_tth = sorted(valid, key=lambda s: -s["sigma_strain_pix"])[:top_n]
    for s in top_tth:
        h, k, l = s["hkl"]
        ax_scatter.annotate(
            f"({h}{k}{l})",
            xy=(s["tth"], s["sigma_strain_pix"]),
            xytext=(3, 2), textcoords="offset points",
            fontsize=5, color="#aaccff",
        )

    ax_scatter.legend(fontsize=6, framealpha=0.2, facecolor=BG, labelcolor="white",
                      loc="upper right")

    # ── Panel C: Jacobian heat-map ────────────────────────────────────────────
    if has_jac and ax_jac is not None:
        ax_jac.set_facecolor(BG)
        ax_jac.set_title(
            "Strain sensitivity  |∂pix/∂εᵢⱼ|  (px per unit strain)",
            color=FG, fontsize=8,
        )
        ax_jac.tick_params(colors="#7788aa", labelsize=6)
        for sp in ax_jac.spines.values():
            sp.set_edgecolor("#1a1f2e")

        top_hkl = [s["hkl"] for s in top_spots]
        # Build heatmap matrix: rows = spots, cols = Voigt components
        # Cell value = RMS of the two rows of J (xcam and ycam sensitivity)
        heat = np.zeros((len(top_hkl), 6))
        for ri, hkl in enumerate(top_hkl):
            J = jacobians.get(hkl)
            if J is not None:
                heat[ri] = np.sqrt(0.5 * (J[0] ** 2 + J[1] ** 2))

        im_j = ax_jac.imshow(
            heat,
            aspect="auto",
            cmap="YlOrRd",
            interpolation="nearest",
        )
        ax_jac.set_xticks(range(6))
        ax_jac.set_xticklabels(_VOIGT_LABELS, fontsize=7, color=FG)
        ax_jac.set_yticks(range(len(top_hkl)))
        ax_jac.set_yticklabels(
            [f"({h}{k}{l})" for h, k, l in top_hkl], fontsize=6, color=FG
        )
        ax_jac.set_xlabel("Voigt strain component", color=FG, fontsize=7)

        # Annotate cells with the value
        for ri in range(len(top_hkl)):
            for ci in range(6):
                v = heat[ri, ci]
                if v > 0.01:
                    ax_jac.text(
                        ci, ri, f"{v:.1f}",
                        ha="center", va="center",
                        fontsize=5,
                        color="white" if v > heat.max() * 0.6 else "black",
                    )

        cb_j = fig.colorbar(im_j, ax=ax_jac, fraction=0.05, pad=0.02)
        cb_j.set_label("px / unit strain", color="#7788aa", fontsize=6)
        cb_j.ax.tick_params(colors="#7788aa", labelsize=5)
        cb_j.outline.set_edgecolor("#1a1f2e")

    fig.suptitle("Laue spot broadening due to elastic strain", color=FG,
                 fontsize=11, y=1.01)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Strain broadening plot saved → {out_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# COMPARE TWO SPOT TABLES
# ─────────────────────────────────────────────────────────────────────────────

def plot_compare_spots(
    spots_a,
    spots_b,
    *,
    space: str = "angles",
    label_a: str = "Set A",
    label_b: str = "Set B",
    E_MIN_eV: float = 5_000,
    E_MAX_eV: float = 27_000,
    n_label: int = 8,
    out_path: str | None = "compare_spots.png",
):
    """
    Overlay two spot tables on a single axes for direct comparison.

    Args:
        spots_a, spots_b (list of dict): Spot dicts from :func:`~nrxrdct.laue.simulation.simulate_laue` or
            compatible sources.  Each dict must contain:

        * `'tth'`         – 2θ in degrees
        * `'chi'`         – χ in degrees
        * `'pix'`         – `(col, row)` pixel coordinate on the detector
        * `'E'`           – photon energy in eV
        * `'hkl'`         – Miller indices tuple `(h, k, l)`
        * `'intensity'`   – normalised intensity [0, 1]
        * `'is_superlattice'` – bool

        space (`'angles'` | `'detector'`): `'angles'`   – x-axis = 2θ (degrees), y-axis = χ (degrees).
            `'detector'` – x-axis = column pixel,  y-axis = row pixel.

        label_a, label_b (str): Legend labels for the two spot sets.

        E_MIN_eV, E_MAX_eV (float): Energy range for the shared colour-map.

        n_label (int): Number of strongest spots in each set to annotate with (hkl).

        out_path (str or None): File path for the saved PNG.  `None` → do not save.

    Returns:
        fig (matplotlib.figure.Figure):
"""
    if space not in ("angles", "detector"):
        raise ValueError(f"space must be 'angles' or 'detector', got {space!r}")

    # ── colour maps: Set A = cool blues, Set B = warm oranges ────────────────
    CMAP_A = "Blues"
    CMAP_B = "Oranges"
    E_norm = mcolors.Normalize(vmin=E_MIN_eV / 1e3, vmax=E_MAX_eV / 1e3)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors="#7788aa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1f2e")

    def _xy(s):
        if space == "angles":
            return s["tth"], s["chi"]
        else:
            pix = s.get("pix")
            if pix is None:
                return None, None
            return float(pix[0]), float(pix[1])

    def _draw(spots, cmap, marker):
        valid = [s for s in spots if _xy(s)[0] is not None]
        if not valid:
            return

        fund = [s for s in valid if not s.get("is_superlattice")]
        super_ = [s for s in valid if s.get("is_superlattice")]

        for subset, mk, alpha in [(fund, marker, 0.85), (super_, "*", 0.75)]:
            if not subset:
                continue
            xs = [_xy(s)[0] for s in subset]
            ys = [_xy(s)[1] for s in subset]
            Es = [s["E"] / 1e3 for s in subset]
            sz = [max(5, 90 * s["intensity"] ** 0.4) for s in subset]
            ax.scatter(
                xs, ys,
                s=sz,
                c=Es,
                cmap=cmap,
                norm=E_norm,
                alpha=alpha,
                marker=mk,
                linewidths=0.4,
                zorder=3,
            )

        # Annotate n_label strongest fundamental spots
        top = sorted(fund, key=lambda s: -s["intensity"])[:n_label]
        for s in top:
            h, k, l = s["hkl"]
            x, y = _xy(s)
            ax.annotate(
                f"({h}{k}{l})",
                xy=(x, y),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=5.5,
                color="#aaddff",
                alpha=0.9,
            )

    _draw(spots_a, CMAP_A, "o")
    _draw(spots_b, CMAP_B, "D")

    # ── colour bars ───────────────────────────────────────────────────────────
    sm_a = plt.cm.ScalarMappable(cmap=CMAP_A, norm=E_norm)
    sm_b = plt.cm.ScalarMappable(cmap=CMAP_B, norm=E_norm)
    sm_a.set_array([])
    sm_b.set_array([])
    cb_a = fig.colorbar(sm_a, ax=ax, pad=0.01, fraction=0.035, aspect=30)
    cb_b = fig.colorbar(sm_b, ax=ax, pad=0.08, fraction=0.035, aspect=30)
    for cb, lbl in [(cb_a, label_a), (cb_b, label_b)]:
        cb.set_label(f"E  (keV)  [{lbl}]", color="#7788aa", fontsize=7)
        cb.ax.tick_params(colors="#7788aa", labelsize=6)
        cb.outline.set_edgecolor("#1a1f2e")

    # ── legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#5bc8f5",
               markersize=7, label=f"{label_a} – fundamental", linewidth=0),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#5bc8f5",
               markersize=9, label=f"{label_a} – superlattice", linewidth=0),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#f5a845",
               markersize=7, label=f"{label_b} – fundamental", linewidth=0),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#f5a845",
               markersize=9, label=f"{label_b} – superlattice", linewidth=0),
    ]
    ax.legend(
        handles=legend_elements,
        facecolor="#0d1220",
        edgecolor="#333355",
        labelcolor=FG,
        fontsize=7,
        loc="upper right",
    )

    # ── axis labels & title ───────────────────────────────────────────────────
    if space == "angles":
        ax.set_xlabel("2θ  (degrees)", color="#7788aa", fontsize=9)
        ax.set_ylabel("χ  (degrees)", color="#7788aa", fontsize=9)
        title = f"Laue spot comparison — angular space ({label_a} vs {label_b})"
    else:
        ax.set_xlabel("Column  (pixel)", color="#7788aa", fontsize=9)
        ax.set_ylabel("Row  (pixel)", color="#7788aa", fontsize=9)
        ax.invert_yaxis()   # detector rows increase downward
        title = f"Laue spot comparison — detector space ({label_a} vs {label_b})"

    ax.set_title(title, color=FG, fontsize=10, pad=8)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Compare-spots plot saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MEASURED vs SIMULATED COMPARISON
# ─────────────────────────────────────────────────────────────────────────────


def plot_measured_vs_simulated(
    peaklist,
    spots,
    *,
    image=None,
    camera=None,
    n_label: int = 8,
    E_min_eV: float = 5_000,
    E_max_eV: float = 27_000,
    figsize: tuple = (14, 6),
    show_arrows: bool = True,
    max_match_dist: float = 50.0,
    out_path: str | None = None,
):
    """
    Side-by-side comparison of segmented measured spots and simulated Laue spots
    in detector pixel coordinates.

    Args:
        peaklist (ndarray, shape (N, 9)): Output of :func:`~nrxrdct.laue.segmentation.convert_spotsfile2peaklist`.
            Columns: peak_X (col), peak_Y (row), peak_I, peak_fwaxmaj, peak_fwaxmin,
            peak_inclination, Xdev, Ydev, peak_bkg.

        spots (list of dict): Output of :func:`~nrxrdct.laue.simulation.simulate_laue` or compatible
            functions.  Each dict must contain `'pix'` (col, row), `'E'`,
            `'intensity'`, and `'hkl'`.

        image (ndarray of shape (Nv, Nh) or None): Optional raw detector image shown as a background on both panels.
            Displayed with a logarithmic normalisation and the `'inferno'`
            colormap.

        camera (Camera or None): When provided, `camera.Nh` and `camera.Nv` are used to set fixed
            axis limits.  If `None`, limits are derived from the data extent.

        n_label (int): Number of strongest simulated spots to annotate with their (hkl) Miller
            indices on the right panel.

        E_min_eV, E_max_eV (float): Energy range for the simulated-spot colour scale (keV).

        figsize (tuple of (float, float)): Figure size in inches passed to `plt.subplots`.  Default `(14, 6)`.

        show_arrows (bool): When `True` (default), draw displacement arrows on the simulated panel
            from each simulated spot to its nearest measured spot.  Arrows are
            coloured by displacement distance; a mean-displacement annotation and
            colour bar are added automatically.

        max_match_dist (float): Maximum nearest-neighbour distance in pixels below which a simulated
            spot is matched to a measured spot and an arrow is drawn.
            Default `50.0` pixels.

        out_path (str or None): Path to save the PNG figure.  `None` (default) → do not save.

    Returns:
        fig (matplotlib.figure.Figure):
"""
    peaklist = np.asarray(peaklist, dtype=float)

    fig, (ax_m, ax_s) = plt.subplots(
        1, 2, figsize=figsize, sharex=True, sharey=True
    )
    fig.patch.set_facecolor(BG)

    for ax in (ax_m, ax_s):
        ax.set_facecolor(BG)
        ax.tick_params(colors="#7788aa", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a1f2e")
        ax.set_xlabel("Column  (pixel)", color="#7788aa", fontsize=9)
        ax.set_ylabel("Row  (pixel)", color="#7788aa", fontsize=9)

    # ── background image (both panels) ───────────────────────────────────────
    if image is not None:
        img = np.asarray(image, dtype=float)
        vmin = max(img[img > 0].min(), 1.0) if np.any(img > 0) else 1.0
        norm_img = mcolors.LogNorm(vmin=vmin, vmax=img.max())
        for ax in (ax_m, ax_s):
            ax.imshow(
                img,
                origin="upper",
                cmap="inferno",
                norm=norm_img,
                aspect="equal",
                interpolation="nearest",
                alpha=0.6,
                zorder=1,
            )

    # ── measured spots (left panel) ──────────────────────────────────────────
    if peaklist.shape[0] > 0:
        col_m = peaklist[:, 0]
        row_m = peaklist[:, 1]
        I_m = peaklist[:, 2]
        I_pos = np.clip(I_m, 1e-6 * I_m.max() if I_m.max() > 0 else 1e-6, None)
        log_I = np.log10(I_pos)
        log_I_norm = (log_I - log_I.min()) / (log_I.max() - log_I.min() + 1e-12)
        sz_m = np.clip(20 + 200 * log_I_norm**0.4, 5, 300)
        sc_m = ax_m.scatter(
            col_m, row_m,
            s=sz_m,
            c=log_I_norm,
            cmap="viridis",
            vmin=0,
            vmax=1,
            alpha=0.85,
            marker="o",
            linewidths=0.3,
            edgecolors="#ffffff44",
            zorder=3,
        )
        cb_m = fig.colorbar(sc_m, ax=ax_m, pad=0.02, fraction=0.035, aspect=30)
        cb_m.set_label("log₁₀ intensity  (measured, normalised)", color="#7788aa", fontsize=7)
        cb_m.ax.tick_params(colors="#7788aa", labelsize=6)
        cb_m.outline.set_edgecolor("#1a1f2e")

    ax_m.set_title("Measured  (segmentation)", color=FG, fontsize=10, pad=6)

    # ── simulated spots (right panel) ────────────────────────────────────────
    valid = [s for s in spots if s.get("pix") is not None]
    if valid:
        E_norm = mcolors.Normalize(vmin=E_min_eV / 1e3, vmax=E_max_eV / 1e3)

        fund = [s for s in valid if not s.get("is_superlattice")]
        super_ = [s for s in valid if s.get("is_superlattice")]

        for subset, marker, alpha in [(fund, "o", 0.85), (super_, "*", 0.75)]:
            if not subset:
                continue
            col_s = [float(s["pix"][0]) for s in subset]
            row_s = [float(s["pix"][1]) for s in subset]
            Es = [s["E"] / 1e3 for s in subset]
            sz_s = [max(5, 90 * s["intensity"] ** 0.4) for s in subset]
            ax_s.scatter(
                col_s, row_s,
                s=sz_s,
                c=Es,
                cmap="plasma",
                norm=E_norm,
                alpha=alpha,
                marker=marker,
                linewidths=0.4,
                edgecolors="#ffffff33",
                zorder=3,
            )

        # colour bar for energy
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=E_norm)
        sm.set_array([])
        cb_s = fig.colorbar(sm, ax=ax_s, pad=0.02, fraction=0.035, aspect=30)
        cb_s.set_label("E  (keV)  [simulated]", color="#7788aa", fontsize=7)
        cb_s.ax.tick_params(colors="#7788aa", labelsize=6)
        cb_s.outline.set_edgecolor("#1a1f2e")

        # HKL labels on n_label strongest fundamental spots
        top = sorted(fund, key=lambda s: -s["intensity"])[:n_label]
        for s in top:
            h, k, l = s["hkl"]
            ax_s.annotate(
                f"({h}{k}{l})",
                xy=(float(s["pix"][0]), float(s["pix"][1])),
                xytext=(3, 2),
                textcoords="offset points",
                fontsize=5.5,
                color="#aaddff",
                alpha=0.9,
                zorder=4,
            )

    ax_s.set_title("Simulated", color=FG, fontsize=10, pad=6)

    # ── displacement arrows: simulated → nearest measured ────────────────────
    if show_arrows and peaklist.shape[0] > 0 and valid:
        sim_xy = np.array([[float(s["pix"][0]), float(s["pix"][1])] for s in valid])
        meas_xy = peaklist[:, :2]  # (N_meas, 2): col, row

        # Brute-force nearest-neighbour
        diff = sim_xy[:, None, :] - meas_xy[None, :, :]  # (N_sim, N_meas, 2)
        dist_mat = np.linalg.norm(diff, axis=-1)          # (N_sim, N_meas)
        nn_idx = np.argmin(dist_mat, axis=1)               # (N_sim,)
        nn_dist = dist_mat[np.arange(len(sim_xy)), nn_idx] # (N_sim,)

        mask = nn_dist <= max_match_dist
        if mask.any():
            x0 = sim_xy[mask, 0]
            y0 = sim_xy[mask, 1]
            dx = meas_xy[nn_idx[mask], 0] - x0   # measured − simulated
            dy = meas_xy[nn_idx[mask], 1] - y0
            dists = nn_dist[mask]

            vmax_cb = float(np.percentile(dists, 95)) or 1.0
            dist_norm = mcolors.Normalize(vmin=0, vmax=vmax_cb)
            qv = ax_s.quiver(
                x0, y0, dx, dy, dists,
                cmap="RdYlGn_r",
                norm=dist_norm,
                angles="xy",
                scale_units="xy",
                scale=1,
                units="dots",
                width=2.0,
                headwidth=5,
                headlength=6,
                alpha=0.9,
                zorder=5,
            )
            cb_q = fig.colorbar(qv, ax=ax_s, pad=0.14, fraction=0.035, aspect=30)
            cb_q.set_label("Displacement  (pixel)", color="#7788aa", fontsize=7)
            cb_q.ax.tick_params(colors="#7788aa", labelsize=6)
            cb_q.outline.set_edgecolor("#1a1f2e")

            mean_d = dists.mean()
            ax_s.text(
                0.02, 0.02,
                f"mean Δ = {mean_d:.2f} px  (n={mask.sum()})",
                transform=ax_s.transAxes,
                color="#aaddff",
                fontsize=7,
                va="bottom",
                zorder=6,
            )

    # ── axis limits ──────────────────────────────────────────────────────────
    if camera is not None:
        for ax in (ax_m, ax_s):
            ax.set_xlim(0, camera.Nh)
            ax.set_ylim(camera.Nv, 0)   # rows increase downward
    else:
        # derive from data; rows still increase downward
        all_cols, all_rows = [], []
        if peaklist.shape[0] > 0:
            all_cols.extend(peaklist[:, 0].tolist())
            all_rows.extend(peaklist[:, 1].tolist())
        for s in valid:
            all_cols.append(float(s["pix"][0]))
            all_rows.append(float(s["pix"][1]))
        if all_cols:
            margin = 50
            c0, c1 = min(all_cols) - margin, max(all_cols) + margin
            r0, r1 = min(all_rows) - margin, max(all_rows) + margin
            for ax in (ax_m, ax_s):
                ax.set_xlim(c0, c1)
                ax.set_ylim(r1, r0)   # inverted

    fig.suptitle("Measured vs Simulated — detector space", color=FG, fontsize=11, y=1.01)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Measured-vs-simulated plot saved → {out_path}")

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# LAYER SCHEME
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer_scheme(
    stack,
    figsize=(10, 7),
    layer_width=10,
    max_reps=6,
    min_display_frac=0.01,
    ax=None,
    out_path=None,
):
    """
    Render a schematic cross-section of a LayeredCrystal stack oriented in
    the LaueTools lab frame (x = beam direction, z = vertical up).

    The view is the XZ side-plane.  Each layer is drawn as a scaled
    parallelogram whose normal is `stack.n_hat` projected onto XZ.
    Layers too thin to label inside (< `min_display_frac` of total height)
    are annotated with an external callout.

    Buffer layers are drawn at the bottom at a fixed display height (they're
    usually far thicker than everything else and are flagged "not to
    scale"). Every repeating block (see `stack.blocks` /
    :meth:`LayeredCrystal.add_layer`) is drawn above them in sequence, each
    at real relative scale, with a dashed boundary line between the buffer
    section and the first block and between consecutive blocks. Every
    repetition of a block is still drawn as its own coloured slab, but gets
    exactly *one* text label for the whole block (its layer name(s), period,
    and repeat count), centred on the block's full drawn span -- with
    `n_rep` in the tens, labelling every repetition individually would be
    unreadable.

    Args:
        stack (LayeredCrystal):
        figsize ((float, float)):
        layer_width (float): Half-width of the layer slabs in display units.
        max_reps (int): Maximum number of repetitions to draw *per block*.  Blocks with
            more repetitions than this show an ellipsis annotation at their
            own top boundary.
        min_display_frac (float): Layers thinner than this fraction of the drawn stack height have
            their label placed outside with a leader line instead of inside.
        ax (matplotlib Axes, optional): Draw into an existing Axes.  If None a new figure is created.
        out_path (str, optional): Save figure to this path if provided.

    Returns:
        fig, ax
"""
    import matplotlib.patches as mpatches

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    else:
        fig = ax.figure
    ax.set_facecolor(BG)

    # ── Stacking direction projected onto XZ ─────────────────────────────────
    nh = np.asarray(stack.n_hat, dtype=float)
    nh_2d = np.array([nh[0], nh[2]])           # (x, z) display components
    norm2d = np.linalg.norm(nh_2d)
    if norm2d < 1e-6:                           # n_hat purely along y
        nh_2d = np.array([0.0, 1.0])
    else:
        nh_2d /= norm2d
    th_2d = np.array([-nh_2d[1], nh_2d[0]])   # tangent ⊥ nh_2d in XZ

    # Text rotation matching the layer-dividing lines (which run along th_2d).
    # Folded into (-90, 90] so labels stay upright rather than appearing
    # upside-down for steeply tilted stacking directions.
    label_angle_deg = float(np.degrees(np.arctan2(th_2d[1], th_2d[0])))
    if label_angle_deg > 90:
        label_angle_deg -= 180
    elif label_angle_deg <= -90:
        label_angle_deg += 180

    # ── Build layer list (with repetitions) ──────────────────────────────────
    stack._update_offsets()
    blocks = [blk for blk in stack.blocks if blk.layers]
    has_rep = bool(blocks)
    has_buf = bool(stack.buffer_layers)

    if not has_rep and not has_buf:
        if standalone:
            return fig, ax
        return ax

    # Repetition-limited drawn height of each block, in Å
    block_draw = []   # (block, n_reps_draw, drawn_thickness_ang)
    for blk in blocks:
        n_draw = min(blk.n_rep, max_reps)
        block_draw.append((blk, n_draw, n_draw * blk._period))
    total_rep_ang = sum(t for _, _, t in block_draw)

    # ── Scaling ───────────────────────────────────────────────────────────────
    DISP_H = 5.0   # display height reserved for the repeating section (display units)
    W      = layer_width

    if has_rep and total_rep_ang > 1e-9:
        scale = DISP_H / total_rep_ang          # Å → display units (repeating blocks only)
        # Buffer layers are drawn at a fixed height = thickest repeated layer,
        # shared across all blocks so buffer layers read consistently.
        max_rep_ang = max(lyr.thickness for blk, _, _ in block_draw for lyr in blk.layers)
        buf_disp_h  = max_rep_ang * scale
    else:
        # No repeating blocks — fall back to normal scaling for buffer-only stacks
        scale      = DISP_H / stack._buffer_thickness
        buf_disp_h = None   # use real thickness

    # ── Build (layer, s0_disp, s1_disp, is_buffer) list ───────────────────────
    # Display positions are computed independently for buffers (fixed height)
    # and repeating-block layers (real scale, shared across all blocks).
    # Every repetition of a block is drawn as its own coloured parallelogram,
    # but individual repetitions/sub-layers never get their own text -- with
    # n_rep in the tens that would be a wall of duplicate, overlapping
    # labels.  Instead each block gets exactly *one* label, centred on its
    # full drawn span (see `block_labels` below); only buffer layers (which
    # never repeat) keep their per-layer inline label.
    layers_to_draw = []      # (layer, s0, s1, is_buffer)
    boundary_lines = []      # display positions s where a dashed section boundary is drawn
    ellipsis_notes = []      # (s_position, text) for blocks clipped by max_reps
    block_labels = []        # (center_s, span_disp, text, color) -- one per block

    # Buffer layers stacked from z_disp=0, each at a fixed display height
    z_disp = 0.0
    for layer in stack.buffer_layers:
        h = buf_disp_h if buf_disp_h is not None else layer.thickness * scale
        layers_to_draw.append((layer, z_disp, z_disp + h, True))
        z_disp += h

    if has_buf and has_rep:
        boundary_lines.append(z_disp)   # buffer → first repeating block

    # Repeating blocks, each at real scale, stacked in sequence
    for bi, (blk, n_draw, _drawn_ang) in enumerate(block_draw):
        block_start_disp = z_disp
        for _rep in range(n_draw):
            for layer, z0_local in zip(blk.layers, blk._z_offsets):
                s0 = z_disp + z0_local * scale
                s1 = s0 + layer.thickness * scale
                layers_to_draw.append((layer, s0, s1, False))
            z_disp += blk._period * scale

        block_names = list(dict.fromkeys(lyr.label for lyr in blk.layers))
        period_nm = blk._period / 10.0
        block_text = (
            f"{' + '.join(block_names)}\n"
            f"{period_nm:.1f} nm period  ×{blk.n_rep}"
        )
        block_labels.append(
            # colour resolved below, once cmap exists -- store the label
            # name as a placeholder in its slot for now.
            [(block_start_disp + z_disp) * 0.5, z_disp - block_start_disp,
             block_text, blk.layers[0].label]
        )

        if blk.n_rep > max_reps:
            ellipsis_notes.append((z_disp, f"⋮  ({blk.n_rep} repetitions total)"))
        if bi < len(block_draw) - 1:
            boundary_lines.append(z_disp)   # this block → the next one

    total_disp_h = z_disp

    # ── Colour map (unique layer labels) ─────────────────────────────────────
    unique_labels = list(dict.fromkeys(t[0].label for t in layers_to_draw))
    palette = plt.cm.Set2(np.linspace(0.0, 0.85, max(len(unique_labels), 1)))
    cmap = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}
    # Resolve each block label's representative colour now that cmap exists.
    block_labels = [
        (center, span, text, cmap[first_label])
        for center, span, text, first_label in block_labels
    ]

    # ── Draw section boundary markers (buffer→block and block→block) ─────────
    for s_bound in boundary_lines:
        pt_l = s_bound * nh_2d - W * th_2d
        pt_r = s_bound * nh_2d + W * th_2d
        ax.plot(
            [pt_l[0], pt_r[0]], [pt_l[1], pt_r[1]],
            color="#ffdd55", linewidth=1.8, linestyle="--", zorder=4,
        )

    # ── Draw layers ───────────────────────────────────────────────────────────
    callout_labels = []   # [(center_xy, text, color)] for thin-layer callouts

    for layer, s0, s1, is_buffer in layers_to_draw:
        color = cmap[layer.label]

        # Parallelogram corners in (x_display, z_display)
        c0 = s0 * nh_2d - W * th_2d
        c1 = s0 * nh_2d + W * th_2d
        c2 = s1 * nh_2d + W * th_2d
        c3 = s1 * nh_2d - W * th_2d
        edge_lw    = 1.2 if is_buffer else 0.7
        edge_color = "#ffdd55" if is_buffer else "white"
        poly = plt.Polygon(
            [c0, c1, c2, c3], closed=True,
            facecolor=color, edgecolor=edge_color, linewidth=edge_lw,
            alpha=0.88, zorder=2,
        )
        ax.add_patch(poly)

        if not is_buffer:
            # Repeating-block layers get one label for the whole block
            # (below), not one per individual layer/repetition.
            continue

        cx = ((s0 + s1) * 0.5) * nh_2d   # centre of parallelogram
        ds = s1 - s0
        thick_nm = layer.thickness / 10.0
        thick_um = layer.thickness / 1e4
        thick_str = f"{thick_um:.2f} µm" if thick_um >= 0.1 else f"{thick_nm:.1f} nm"
        label_str = f"{layer.label}\n{thick_str}\n(not to scale)"

        if ds >= min_display_frac * DISP_H:
            ax.text(
                cx[0], cx[1], label_str,
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                rotation=label_angle_deg, rotation_mode="anchor",
                color="black", zorder=3, clip_on=True,
            )
        else:
            # Too thin to label inside — queue a callout
            callout_labels.append((cx, label_str, color))

    # ── One label per repeating block, centred on its full drawn span ────────
    for center_s, span_disp, block_text, block_color in block_labels:
        cx = center_s * nh_2d
        if span_disp >= min_display_frac * DISP_H:
            ax.text(
                cx[0], cx[1], block_text,
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                rotation=label_angle_deg, rotation_mode="anchor",
                color="black", zorder=3, clip_on=True,
            )
        else:
            callout_labels.append((cx, block_text, block_color))

    # Callouts for thin layers (placed to the right of the stack)
    if callout_labels:
        right_edge = W * th_2d   # XZ vector to the right edge of the slab
        offset_step = 0.45
        for i, (cx, txt, col) in enumerate(callout_labels):
            tip   = cx + right_edge + np.array([0.15, 0.0])
            label_xy = tip + np.array([0.35 + i * 0.0, i * offset_step - len(callout_labels) * offset_step * 0.4])
            ax.annotate(
                txt,
                xy=tip, xytext=label_xy,
                fontsize=7, color="black", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=col, ec="white", lw=0.5),
                arrowprops=dict(arrowstyle="-", color="white", lw=0.8),
                zorder=4,
            )

    # Clipped-reps markers -- one per block that exceeded max_reps
    for s_pos, text in ellipsis_notes:
        tip_pos = s_pos * nh_2d
        ax.text(
            tip_pos[0] + 0.05, tip_pos[1] + 0.12,
            text,
            color=FG, fontsize=8, va="bottom", zorder=4,
        )

    # ── Incident beam arrow ───────────────────────────────────────────────────
    # The beam (+x) hits the last layer (largest z0 = surface-facing end).
    surf_ctr  = total_disp_h * nh_2d
    beam_tip  = surf_ctr - 0.2 * nh_2d              # slightly inset
    beam_tail = beam_tip + np.array([-(W + 1.5), 0.0])   # start outside the slab
    ax.annotate(
        "", xy=beam_tip, xytext=beam_tail,
        arrowprops=dict(arrowstyle="->", color=COL_DB, lw=2.5, mutation_scale=20),
        zorder=5,
    )
    mid_beam = 0.5 * (beam_tip + beam_tail)
    ax.text(
        mid_beam[0], mid_beam[1] + 0.25,
        "incident beam  (+x)",
        color=COL_DB, fontsize=8, ha="center", va="bottom",
    )

    # ── Beam path inside sample (dashed red) ─────────────────────────────────
    beam_exit = beam_tip + np.array([2.0 * W, 0.0])
    ax.plot(
        [beam_tip[0], beam_exit[0]], [beam_tip[1], beam_exit[1]],
        color="red", lw=1.5, linestyle="--", zorder=4,
    )

    # ── Surface-normal arrow ──────────────────────────────────────────────────
    n_base = surf_ctr
    n_tip  = surf_ctr + 1.2 * nh_2d
    ax.annotate(
        "", xy=n_tip, xytext=n_base,
        arrowprops=dict(arrowstyle="->", color="white", lw=1.8, mutation_scale=12),
        zorder=5,
    )
    nh_label = (
        r"$\hat{n}$"
        f"  [{nh[0]:+.2f}, {nh[1]:+.2f}, {nh[2]:+.2f}]"
        "\n(stacking direction)"
    )
    ax.text(
        n_tip[0] + 0.1 * nh_2d[0],
        n_tip[1] + 0.12,
        nh_label,
        color="white", fontsize=8, ha="center", va="bottom",
    )

    # ── Surface / substrate edge labels ──────────────────────────────────────
    surf_edge_r = surf_ctr + (W + 0.12) * th_2d
    subs_edge_r = 0 * nh_2d + (W + 0.12) * th_2d
    ax.text(surf_edge_r[0], surf_edge_r[1], "surface ▶",
            color="white", fontsize=9.5, fontweight="bold", va="center", ha="left",
            rotation=label_angle_deg, rotation_mode="anchor")
    ax.text(subs_edge_r[0], subs_edge_r[1], "substrate ▶",
            color="white", fontsize=9.5, fontweight="bold", va="center", ha="left",
            rotation=label_angle_deg, rotation_mode="anchor")

    # ── Lab frame axes (bottom-left corner) ──────────────────────────────────
    ax_len  = 1.5
    # Find a comfortable lower-left position
    all_pts = np.array([s * nh_2d + sign * W * th_2d
                        for s in [0, total_disp_h]
                        for sign in [-1, 1]])
    x_min = all_pts[:, 0].min() - 4.0
    z_min = all_pts[:, 1].min() - 0.5
    orig  = np.array([x_min, z_min])

    def _axis_arrow(direction, color, label, label_offset):
        end = orig + ax_len * direction
        ax.annotate(
            "", xy=end, xytext=orig,
            arrowprops=dict(arrowstyle="->", color=color, lw=2.0, mutation_scale=16),
            zorder=6,
        )
        lp = end + label_offset
        ax.text(lp[0], lp[1], label, color=color, fontsize=9,
                fontweight="bold", ha="center", va="center")

    _axis_arrow(np.array([1.0, 0.0]), "#4fc3f7", "x\n(beam)",   np.array([0.45, 0.0]))
    _axis_arrow(np.array([0.0, 1.0]), "#ff9f43", "z\n(up)",     np.array([0.0,  0.45]))

    # y-axis: out-of-plane, shown as ⊙
    ax.plot(*orig, "o", color="#88cc88", ms=8, zorder=6)
    ax.plot(*orig, ".", color="#88cc88", ms=3, zorder=7)
    ax.text(orig[0] - 0.45, orig[1], "y\n(out)", color="#88cc88",
            fontsize=9, fontweight="bold", ha="center", va="center")

    # ── n_hat numeric annotation ──────────────────────────────────────────────
    nh_str = (f"$\\hat{{n}}$ = [{nh[0]:+.3f},  {nh[1]:+.3f},  {nh[2]:+.3f}]"
              + ("  (XZ projection)" if abs(nh[1]) > 1e-3 else ""))
    ax.text(
        0.5, 0.01, nh_str,
        transform=ax.transAxes, color="#8899bb", fontsize=8,
        ha="center", va="bottom",
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(facecolor=cmap[lbl], edgecolor="white", lw=0.5, label=lbl)
        for lbl in unique_labels
    ]
    ax.legend(
        handles=handles, loc="lower right", fontsize=8.5,
        framealpha=0.45, facecolor="#1a1f2e", edgecolor="#3a3f4e",
        labelcolor="white", handlelength=1.2, handleheight=0.9,
    )

    # ── Title ─────────────────────────────────────────────────────────────────
    total_um  = stack.total_thickness / 1e4
    n_clipped = sum(1 for blk, _, _ in block_draw if blk.n_rep > max_reps)
    reps_note = (f"  [{n_clipped} block(s) capped at {max_reps} reps for display]"
                 if n_clipped else "")
    ax.set_title(
        f"{stack.name}   —   total thickness {total_um:.3f} µm{reps_note}",
        color=FG, fontsize=9, pad=7,
    )

    # ── Final styling ─────────────────────────────────────────────────────────
    ax.set_aspect("equal")
    ax.axis("off")
    ax.autoscale_view()

    fig.patch.set_facecolor(BG)
    if standalone:
        fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Layer scheme saved → {out_path}")
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# LAYERED STACK SPOT MAP
# ─────────────────────────────────────────────────────────────────────────────

#: Marker cycle for phases — each new unique `phase_label` gets the next one.
_PHASE_MARKERS = ["o", "s", "^", "D", "v", "p", "h", "X", "<", ">", "*"]

#: Per-phase base colour palette (one hue per phase, used for the Bragg peaks).
_PHASE_PALETTES = [
    "Blues_r",
    "Oranges_r",
    "Greens_r",
    "Purples_r",
    "RdPu_r",
    "YlOrBr_r",
    "GnBu_r",
    "PuRd_r",
]


def plot_laue_stack_spots(
    spots,
    *,
    space: str = "angles",
    n_label: int = 5,
    size_scale: float = 80.0,
    min_size: float = 8.0,
    show_divergence: bool = True,
    divergence_nsigma: float = 2.0,
    image=None,
    log_scale: bool = True,
    image_alpha: float = 0.6,
    figsize=(9, 7),
    ax=None,
    out_path: str | None = "laue_stack_spots.png",
):
    """
    Visualise the spot table from :func:`~nrxrdct.laue.simulate_laue_stack`.

    Each phase (unique `phase_label`) gets a distinct **marker shape**.
    Within a phase, the **marker colour** encodes the satellite / fringe order:

    * `satellite_order = 0`   — Bragg peak:  brightest colour of the phase palette.
    * `satellite_order = ±m`  — fringe / superlattice satellite:  progressively
      darker / more saturated shades along the same palette (negative and positive
      orders share the same colour sequence, distinguished by the legend).

    Marker *size* scales with normalised intensity.

    Args:
        spots (list[dict]): Spot list returned by :func:`~nrxrdct.laue.simulate_laue_stack`.
            Required keys: `'phase_label'`, `'satellite_order'`, `'tth'`,
            `'chi'`, `'pix'`, `'intensity'`.
        space (`'angles'` | `'detector'`): Coordinate space to plot in.

        * `'angles'`   — x = 2θ (°), y = χ (°).
        * `'detector'` — x = column pixel, y = row pixel.
        n_label (int): Number of the strongest spots to annotate with `(hkl)` labels.
        size_scale (float): Maximum marker area (`s` kwarg in `ax.scatter`).
        min_size (float): Minimum marker area so that weak spots remain visible.
        show_divergence (bool): When `True` (default), draw a divergence-broadening ellipse around
            each spot that carries the keys added by
            :func:`~nrxrdct.laue.beam_divergence_ellipses`.  Ellipses are shown
            at the `divergence_nsigma`-σ confidence level.  No ellipses are
            drawn for spots with zero broadening (i.e. when the simulation was
            run without divergence parameters).
        divergence_nsigma (float): Size of the drawn ellipse in units of σ.  Default `2.0` (≈ 86 %
            enclosed probability in 2-D).
        image (ndarray of shape (Nv, Nh) or None): Raw detector image shown as a background.
            Only used when `space='detector'`.  `None` → no background.
        log_scale (bool): Apply logarithmic normalisation to *image* before display.
            Default `True`.
        image_alpha (float): Opacity of the background image.  Default `0.6`.
        figsize ((float, float)): Figure size in inches (ignored if *ax* is supplied).
        ax (matplotlib.axes.Axes, optional): Draw into an existing Axes; if *None* a new figure is created.
        out_path (str or None): Save the figure to this path.  `None` → do not save.

    Returns:
        fig (matplotlib.figure.Figure):
        ax (matplotlib.axes.Axes):
"""
    if not spots:
        raise ValueError("spots list is empty")

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(BG)
    else:
        fig = ax.figure

    ax.set_facecolor(BG)
    ax.tick_params(colors="#7788aa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1f2e")

    # ── Coordinate helper ─────────────────────────────────────────────────────
    def _xy(s):
        if space == "angles":
            return float(s["tth"]), float(s["chi"])
        pix = s.get("pix")
        if pix is None:
            return None, None
        return float(pix[0]), float(pix[1])

    # ── Gather phase / order metadata ─────────────────────────────────────────
    # Preserve insertion order (phase order as they appear in the spot list)
    phases = list(dict.fromkeys(s["phase_label"] for s in spots))
    all_orders = sorted({s["satellite_order"] for s in spots})

    phase_marker = {ph: _PHASE_MARKERS[i % len(_PHASE_MARKERS)]
                    for i, ph in enumerate(phases)}

    # Build colour lookup: (phase, order) → RGBA
    # For each phase use a sequential palette; index 0 = Bragg, higher = fringes.
    # Negative and positive orders with the same |m| share the same colour so
    # that the legend stays compact.
    abs_orders_sorted = sorted({abs(m) for m in all_orders})

    def _phase_color(phase_idx, abs_order):
        """Map phase + |satellite_order| → colour from the phase palette."""
        cmap_name = _PHASE_PALETTES[phase_idx % len(_PHASE_PALETTES)]
        cmap_fn = plt.get_cmap(cmap_name)
        n = len(abs_orders_sorted)
        if n == 1:
            # Only one level → use brightest (low value in _r maps)
            return cmap_fn(0.15)
        # Map index 0 → 0.15 (bright), index n-1 → 0.85 (dark/saturated)
        idx = abs_orders_sorted.index(abs_order)
        t = 0.15 + 0.70 * idx / (n - 1)
        return cmap_fn(t)

    phase_order_color = {
        (ph, abs_m): _phase_color(i, abs_m)
        for i, ph in enumerate(phases)
        for abs_m in abs_orders_sorted
    }

    # ── Background detector image ─────────────────────────────────────────────
    if image is not None and space == "detector":
        img = np.asarray(image, dtype=float)
        if log_scale:
            vmin = max(img[img > 0].min(), 1.0) if np.any(img > 0) else 1.0
            norm_img = mcolors.LogNorm(vmin=vmin, vmax=img.max())
        else:
            norm_img = mcolors.Normalize(vmin=img.min(), vmax=img.max())
        ax.imshow(
            img,
            origin="upper",
            cmap="inferno",
            norm=norm_img,
            aspect="equal",
            interpolation="nearest",
            alpha=image_alpha,
            zorder=1,
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    # Group by (phase, satellite_order) so each group is one scatter call
    from collections import defaultdict
    groups = defaultdict(list)
    for s in spots:
        xy = _xy(s)
        if xy[0] is None:
            continue
        groups[(s["phase_label"], s["satellite_order"])].append((xy, s))

    for (phase, order), members in sorted(groups.items(),
                                          key=lambda kv: (phases.index(kv[0][0]),
                                                          kv[0][1])):
        xs = [m[0][0] for m in members]
        ys = [m[0][1] for m in members]
        sizes = [max(min_size, size_scale * m[1]["intensity"]) for m in members]
        marker = phase_marker[phase]
        color = phase_order_color[(phase, abs(order))]
        ax.scatter(xs, ys, s=sizes, c=[color], marker=marker,
                   linewidths=0.4, edgecolors="white", alpha=0.90, zorder=3)

    # ── Divergence ellipses ───────────────────────────────────────────────────
    valid = [s for s in spots if _xy(s)[0] is not None]
    if show_divergence:
        frame = "tth_chi" if space == "angles" else "detector"
        div_spots, div_xs, div_ys = [], [], []
        for s in valid:
            x, y = _xy(s)
            if x is not None:
                div_spots.append(s)
                div_xs.append(x)
                div_ys.append(y)
        if div_spots:
            cols = [phase_order_color[(s["phase_label"], abs(s["satellite_order"]))]
                    for s in div_spots]
            _draw_divergence_ellipses(ax, div_spots,
                                      np.array(div_xs), np.array(div_ys),
                                      frame, divergence_nsigma, cols)

    # ── Annotate strongest spots ───────────────────────────────────────────────
    top_n = sorted(valid, key=lambda s: s["intensity"], reverse=True)[:n_label]
    for s in top_n:
        x, y = _xy(s)
        h, k, l = s["hkl"]
        m = s["satellite_order"]
        lbl = f"({h}{k}{l})" if m == 0 else f"({h}{k}{l})\nm={m:+d}"
        ax.text(x + 0.1, y + 0.1, lbl, color=FG, fontsize=6, va="bottom",
                zorder=5)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = []

    # Phase markers (shape legend)
    for ph in phases:
        c = phase_order_color[(ph, 0)]          # Bragg colour for this phase
        h = Line2D([0], [0], linestyle="none", marker=phase_marker[ph],
                   color=c, markeredgecolor="white", markeredgewidth=0.5,
                   markersize=7, label=ph)
        legend_handles.append(h)

    # Satellite-order colours (shared across all phases; use phase 0 colours)
    ph0 = phases[0]
    for abs_m in abs_orders_sorted:
        if abs_m == 0:
            lbl = "Bragg  (m=0)"
        elif abs_m == 1:
            lbl = "satellite  m=±1"
        else:
            lbl = f"satellite  m=±{abs_m}"
        c = phase_order_color[(ph0, abs_m)]
        h = Line2D([0], [0], linestyle="none", marker="o",
                   color=c, markeredgecolor="white", markeredgewidth=0.5,
                   markersize=7, label=lbl)
        legend_handles.append(h)

    ax.legend(
        handles=legend_handles, loc="upper right", fontsize=7.5,
        framealpha=0.5, facecolor="#1a1f2e", edgecolor="#3a3f4e",
        labelcolor=FG, handlelength=1.0, handleheight=1.0,
        title="phase  /  order", title_fontsize=7.5,
    )

    # ── Axis labels & title ───────────────────────────────────────────────────
    if space == "angles":
        ax.set_xlabel("2θ  (°)", color="#7788aa", fontsize=8)
        ax.set_ylabel("χ  (°)", color="#7788aa", fontsize=8)
    else:
        ax.set_xlabel("column  (px)", color="#7788aa", fontsize=8)
        ax.set_ylabel("row  (px)", color="#7788aa", fontsize=8)

    n_phases = len(phases)
    n_sats = len([m for m in all_orders if m != 0])
    ax.set_title(
        f"Laue stack — {n_phases} phase{'s' if n_phases != 1 else ''}  |  "
        f"{len(valid)} spots  |  {n_sats} satellite orders",
        color=FG, fontsize=9, pad=6,
    )

    if standalone:
        fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Stack spot map saved → {out_path}")
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# SHARED DIVERGENCE-ELLIPSE HELPER
# ─────────────────────────────────────────────────────────────────────────────


def _draw_divergence_ellipses(ax, spots, xs, ys, frame, nsigma, colors):
    """
    Add per-spot broadening ellipses to *ax*.

    Uses the keys written by :func:`~nrxrdct.laue.beam_divergence_ellipses`:
    angle-space keys for `frame='tth_chi'`, pixel-space keys for
    `frame='detector'`.  Spots with zero broadening are silently skipped.

    Args:
        ax (Axes):
        spots (list of spot dicts (aligned with xs/ys)):
        xs, ys (1-D arrays of centre coordinates in display units):
        frame (`'tth_chi'` | `'detector'`):
        nsigma (float): ellipse scale in σ units
        colors (list or str): edge colours, one per spot or a single string
"""
    from matplotlib.patches import Ellipse

    if frame == "tth_chi":
        _maj = "sigma_major_ang_deg"
        _min = "sigma_minor_ang_deg"
        _ang = "ellipse_angle_ang_deg"
    else:
        _maj = "sigma_major_px"
        _min = "sigma_minor_px"
        _ang = "ellipse_angle_px_deg"

    single_color = isinstance(colors, str)
    for i, (s, x, y) in enumerate(zip(spots, xs, ys)):
        sig_maj = s.get(_maj, 0.0)
        if not sig_maj > 0.0:
            continue
        col = colors if single_color else colors[i]
        ax.add_patch(Ellipse(
            xy=(x, y),
            width=2.0 * nsigma * sig_maj,
            height=2.0 * nsigma * s.get(_min, sig_maj),
            angle=s.get(_ang, 0.0),
            linewidth=0.7,
            edgecolor=col,
            facecolor="none",
            alpha=0.55,
            zorder=2,
        ))


# ─────────────────────────────────────────────────────────────────────────────
# SHARED HOVER-TOOLTIP HELPERS
# ─────────────────────────────────────────────────────────────────────────────


def _fmt_hkl(h, k, l):
    """Format Miller indices with overbars for negatives."""
    def _idx(n):
        return f"{abs(n)}\u0305" if n < 0 else str(n)
    return f"({_idx(h)} {_idx(k)} {_idx(l)})"


def _spot_label(s):
    """Build the multi-line tooltip string for a single spot dict."""
    h, k, l = s["hkl"]

    sat_order = s.get("satellite_order", None)
    if sat_order is not None and sat_order != 0:
        refl_type = f"satellite  m={sat_order:+d}"
    else:
        refl_type = "Bragg peak"

    phase = s.get("phase_label", None)

    W = 7
    sep = "\u2500" * 22
    lines = [f" hkl   {_fmt_hkl(h, k, l)}"]
    if phase is not None:
        lines.append(f" phase  {phase}")
    lines += [
        sep,
        f" {'2θ':<{W}} {s['tth']:.3f}°",
        f" {'χ':<{W}} {s['chi']:.3f}°",
        f" {'E':<{W}} {s['E']:.1f} eV",
        f" {'I':<{W}} {s['intensity']:.4f}",
        sep,
        f" {'type':<{W}} {refl_type}",
    ]
    return "\n".join(lines)


def _attach_hover_tooltip(fig, ax, spots, tths, chis):
    """
    Attach a hover tooltip to *ax* for the given spot list.

    *tths* and *chis* must be numpy arrays whose indices correspond 1-to-1
    with *spots*.  The tooltip is implemented via `motion_notify_event`
    and requires no extra dependencies beyond matplotlib.
"""
    annot = ax.annotate(
        "",
        xy=(0, 0),
        xytext=(14, 14),
        textcoords="offset points",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="#1a1f2e",
            edgecolor="#4fc3f7",
            linewidth=0.8,
            alpha=0.92,
        ),
        arrowprops=dict(arrowstyle="->", color="#4fc3f7", lw=0.8),
        color=FG,
        fontsize=7.5,
        fontfamily="monospace",
        visible=False,
        zorder=10,
    )

    _hit_radius_pts = 8.0

    def _on_motion(event):
        if event.inaxes is not ax:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        xy_disp = ax.transData.transform(np.column_stack([tths, chis]))
        mouse_disp = np.array([event.x, event.y])
        dists = np.linalg.norm(xy_disp - mouse_disp, axis=1)

        pts_to_px = fig.dpi / 72.0
        threshold_px = _hit_radius_pts * pts_to_px

        idx = int(np.argmin(dists))
        if dists[idx] > threshold_px:
            if annot.get_visible():
                annot.set_visible(False)
                fig.canvas.draw_idle()
            return

        annot.xy = (tths[idx], chis[idx])
        annot.set_text(_spot_label(spots[idx]))

        x_frac = (tths[idx] - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        offset_x = -14 - 140 if x_frac > 0.75 else 14
        annot.xyann = (offset_x, 14)

        if not annot.get_visible():
            annot.set_visible(True)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_motion)
    return annot


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE 2θ / χ SCATTER WITH HOVER TOOLTIPS
# ─────────────────────────────────────────────────────────────────────────────


def plot_interactive_tth_chi(
    spots,
    *,
    color_by: str = "energy",
    i_thresh: float = 0.01,
    size_scale: float = 120.0,
    min_size: float = 10.0,
    figsize=(9, 7),
    out_path: str | None = None,
):
    """
    Interactive 2θ / χ scatter plot with per-spot hover tooltips.

    Hovering the mouse over a marker displays a tooltip containing:

    * Miller indices `(hkl)`
    * 2θ and χ in degrees
    * Photon energy (eV)
    * Normalised intensity
    * Whether the reflection is a **Bragg peak**, **superlattice / superstructure**
      reflection, or a **satellite fringe** (with order *m*)
    * Phase label (when present)

    The function works with spot lists from all three simulation functions:
    :func:`~nrxrdct.laue.simulate_laue`,
    :func:`~nrxrdct.laue.simulate_laue_stack`, and
    :func:`~nrxrdct.laue.simulate_mixed_phases`.

    Args:
        spots (list[dict]): Spot list from any `simulate_laue*` function.  Required keys:
            `'tth'`, `'chi'`, `'hkl'`, `'E'`, `'intensity'`.
            Optional: `'satellite_order'`, `'is_superlattice'`,
            `'phase_label'`.
        i_thresh (float): Minimum intensity threshold as a fraction of the brightest **Bragg
            peak** (`satellite_order == 0`).  Spots with
            `intensity < i_thresh * I_bragg_max` are dropped before plotting.
            Default: `0.01` (1 % of the strongest Bragg peak).
            Pass `0.0` to show all spots.
        color_by (`'energy'` | `'intensity'` | `'phase'`): Quantity mapped to spot colour:

        * `'energy'`    — photon energy (plasma colormap)
        * `'intensity'` — normalised intensity (viridis colormap)
        * `'phase'`     — phase label (tab10; requires `'phase_label'` key)
        size_scale (float): Maximum marker area (`s` in `scatter`).
        min_size (float): Minimum marker area so that weak spots remain visible.
        figsize ((float, float)): Figure size in inches.
        out_path (str or None): If given, save a **static** PNG snapshot on figure close.
            `None` (default) → do not save.

    Returns:
        fig (matplotlib.figure.Figure):
        ax (matplotlib.axes.Axes):

    Note:
    The interactive hover is implemented with matplotlib's built-in event
    system (no extra dependencies).  Call `plt.show()` after this function
    to display the interactive window.
"""
    if not spots:
        raise ValueError("spots list is empty")

    # ── Intensity threshold — relative to brightest Bragg peak ────────────────
    if i_thresh > 0.0:
        bragg_spots = [s for s in spots if s.get("satellite_order", 0) == 0
                       and not s.get("is_superlattice", False)]
        if bragg_spots:
            i_bragg_max = max(s["intensity"] for s in bragg_spots)
        else:
            i_bragg_max = max(s["intensity"] for s in spots)
        cutoff = i_thresh * i_bragg_max
        spots = [s for s in spots if s["intensity"] >= cutoff]
        if not spots:
            raise ValueError(
                f"No spots survive the intensity threshold "
                f"(i_thresh={i_thresh}, cutoff={cutoff:.4f}).  "
                f"Lower i_thresh or pass i_thresh=0.0 to show all."
            )

    # ── Colour mapping ────────────────────────────────────────────────────────
    if color_by == "energy":
        cvals = np.array([s["E"] / 1e3 for s in spots])
        norm = mcolors.Normalize(vmin=cvals.min(), vmax=cvals.max())
        cmap = plt.get_cmap("plasma")
        colors = [cmap(norm(v)) for v in cvals]
        cbar_label = "Energy  (keV)"
    elif color_by == "intensity":
        cvals = np.array([s["intensity"] for s in spots])
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
        cmap = plt.get_cmap("viridis")
        colors = [cmap(norm(v)) for v in cvals]
        cbar_label = "Normalised intensity"
    elif color_by == "phase":
        labels = [s.get("phase_label", "unknown") for s in spots]
        unique_labels = list(dict.fromkeys(labels))
        tab10 = plt.get_cmap("tab10")
        label_color = {lb: tab10(i / max(len(unique_labels) - 1, 1))
                       for i, lb in enumerate(unique_labels)}
        colors = [label_color[lb] for lb in labels]
        cvals = None
        cbar_label = None
    else:
        raise ValueError(f"color_by must be 'energy', 'intensity', or 'phase'")

    sizes = np.array([max(min_size, size_scale * s["intensity"]) for s in spots])
    tths = np.array([s["tth"] for s in spots])
    chis = np.array([s["chi"] for s in spots])

    # ── Figure & axes ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors="#7788aa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1f2e")
    ax.set_xlabel("2θ  (°)", color="#7788aa", fontsize=9)
    ax.set_ylabel("χ  (°)", color="#7788aa", fontsize=9)

    # ── Scatter ───────────────────────────────────────────────────────────────
    ax.scatter(
        tths, chis,
        s=sizes,
        c=colors,
        marker="o",
        linewidths=0.4,
        edgecolors="#ffffff44",
        alpha=0.90,
        zorder=3,
        picker=False,   # we handle picking manually via motion_notify_event
    )

    # ── Colorbar (energy / intensity modes) ───────────────────────────────────
    if cvals is not None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
        cb.set_label(cbar_label, color=FG, fontsize=8)
        cb.ax.tick_params(colors="#7788aa", labelsize=7)
    elif color_by == "phase":
        # Phase legend patches
        from matplotlib.patches import Patch
        handles = [Patch(color=label_color[lb], label=lb)
                   for lb in unique_labels]
        ax.legend(handles=handles, loc="upper right", fontsize=7.5,
                  framealpha=0.5, facecolor="#1a1f2e", edgecolor="#3a3f4e",
                  labelcolor=FG)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f"2θ / χ map  —  {len(spots)} spots   (hover for details)",
        color=FG, fontsize=9, pad=6,
    )

    # ── Hover tooltip ─────────────────────────────────────────────────────────
    _attach_hover_tooltip(fig, ax, spots, tths, chis)

    if out_path:
        import atexit
        atexit.register(
            lambda: fig.savefig(out_path, dpi=150, bbox_inches="tight",
                                facecolor=fig.get_facecolor())
        )

    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# DETECTOR IMAGE → 2θ / χ WARP
# ─────────────────────────────────────────────────────────────────────────────


def warp_image_to_tth_chi(
    image,
    camera,
    *,
    tth_range=None,
    chi_range=None,
    n_tth: int = 600,
    n_chi: int = 600,
    interp_order: int = 1,
):
    """
    Remap a detector image from pixel space into an evenly-spaced 2θ / χ grid.

    For each point (2θ, χ) in the output grid the corresponding scattered
    unit vector is computed, projected back onto the detector via
    :meth:`~nrxrdct.laue.Camera.kf_to_pixel`, and the source image is
    sampled by bilinear interpolation.  Output pixels that map outside the
    detector active area are set to `NaN`.

    Args:
        image (array-like, shape (Nv, Nh)): Detector image in pixel space (e.g. from :meth:`~Camera.render` or
            a real experimental frame loaded as a numpy array).
        camera (Camera): Detector geometry used for the forward/inverse projections.
        tth_range ((float, float), optional): 2θ range in degrees `(tth_min, tth_max)`.  Defaults to the range
            covered by the four detector corners.
        chi_range ((float, float), optional): χ range in degrees `(chi_min, chi_max)`.  Defaults to the range
            covered by the four detector corners.
        n_tth, n_chi (int): Number of output pixels along the 2θ and χ axes.
        interp_order (int): Interpolation order passed to :func:`scipy.ndimage.map_coordinates`
            (0 = nearest, 1 = bilinear (default), 3 = cubic).

    Returns:
        warped (ndarray, shape (n_chi, n_tth)): Remapped image.  NaN where the output pixel falls outside the detector.
        tth_ax (ndarray, shape (n_tth,)): 2θ values of the output columns (degrees).
        chi_ax (ndarray, shape (n_chi,)): χ values of the output rows (degrees).
"""
    from scipy.ndimage import map_coordinates

    image = np.asarray(image, dtype=np.float64)

    # ── Auto-range from detector corners ─────────────────────────────────────
    corner_px = np.array([
        [0,              0],
        [camera.Nh - 1,  0],
        [0,              camera.Nv - 1],
        [camera.Nh - 1,  camera.Nv - 1],
        [camera.xcen,    camera.ycen],
    ])
    ufs = camera.pixel_to_kf(corner_px[:, 0], corner_px[:, 1])
    tths_c = np.degrees(np.arccos(np.clip(ufs[:, 0], -1.0, 1.0)))
    chis_c = np.degrees(np.arctan2(ufs[:, 1], ufs[:, 2] + 1e-17))

    if tth_range is None:
        tth_range = (float(tths_c.min()), float(tths_c.max()))
    if chi_range is None:
        chi_range = (float(chis_c.min()), float(chis_c.max()))

    # ── Output grid ───────────────────────────────────────────────────────────
    tth_ax = np.linspace(tth_range[0], tth_range[1], n_tth)
    chi_ax = np.linspace(chi_range[0], chi_range[1], n_chi)

    # meshgrid: rows = chi, cols = tth  →  shape (n_chi, n_tth)
    TTH, CHI = np.meshgrid(tth_ax, chi_ax)
    tth_r = np.radians(TTH.ravel())
    chi_r = np.radians(CHI.ravel())

    # kf unit vectors in LT frame (x // beam):
    #   kf = [cos 2θ,  sin 2θ · sin χ,  sin 2θ · cos χ]
    sin_tth = np.sin(tth_r)
    kf_arr = np.column_stack([
        np.cos(tth_r),
        sin_tth * np.sin(chi_r),
        sin_tth * np.cos(chi_r),
    ])

    # ── Back-project to detector pixels ──────────────────────────────────────
    xcam, ycam = camera.kf_to_pixel(kf_arr)

    valid = (
        np.isfinite(xcam) & np.isfinite(ycam)
        & (xcam >= 0) & (xcam < camera.Nh)
        & (ycam >= 0) & (ycam < camera.Nv)
    )

    # map_coordinates expects [row, col] = [ycam, xcam]
    coords = np.array([
        np.where(valid, ycam, 0.0),
        np.where(valid, xcam, 0.0),
    ])

    warped_flat = map_coordinates(
        image, coords, order=interp_order, mode="constant", cval=0.0
    )
    warped_flat[~valid] = np.nan

    return warped_flat.reshape(n_chi, n_tth), tth_ax, chi_ax


def plot_tth_chi_overlay(
    image,
    camera,
    spots=None,
    *,
    frame: str = "tth_chi",
    tth_range=None,
    chi_range=None,
    n_tth: int = 600,
    n_chi: int = 600,
    log_scale: bool = True,
    cmap: str = "gray",
    spot_marker: str = "+",
    spot_size: float = 60.0,
    spot_color: str | None = None,
    color_spots_by: str = "phase",
    i_thresh: float = 0.01,
    show_divergence: bool = True,
    divergence_nsigma: float = 2.0,
    figsize=(10, 7),
    out_path: str | None = None,
):
    """
    Overlay simulated spot positions on a detector image.

    Two display frames are available via the *frame* parameter:

    * `'tth_chi'` *(default)* — the detector image is warped from pixel
      coordinates into an evenly-spaced 2θ / χ grid using
      :func:`warp_image_to_tth_chi`, and simulated spots are overlaid at
      their angular positions.
    * `'detector'` — the raw pixel image is displayed without any
      remapping.  Simulated spots are projected to detector pixel coordinates
      (using the `'pix'` key when present, or back-projected from their
      2θ / χ angles via :meth:`~Camera.kf_to_pixel`).

    Hovering over a spot in either frame shows the same tooltip (hkl, 2θ,
    χ, energy, intensity, reflection type, phase).

    Args:
        image (array-like, shape (Nv, Nh)): Detector image in pixel space.
        camera (Camera): Detector geometry.
        spots (list[dict], optional): Spot list from :func:`~nrxrdct.laue.simulate_laue`,
            :func:`~nrxrdct.laue.simulate_laue_stack`, or
            :func:`~nrxrdct.laue.simulate_mixed_phases`.
            Required keys: `'tth'`, `'chi'`.
        frame (`'tth_chi'` | `'detector'`): Coordinate frame for the display (see above).
        tth_range, chi_range ((float, float), optional): Angular range to display (*tth_chi* frame only).
            Defaults to full detector coverage.
        n_tth, n_chi (int): Warp output resolution (*tth_chi* frame only).
        log_scale (bool): Apply `log1p` scaling to the image before display.
        cmap (str): Matplotlib colormap for the image.
        spot_marker (str): Marker style for simulated spots (default `'+'`).
        spot_size (float): Marker size for simulated spots.
        spot_color (str or None): Single colour for all spots.  When `None`, colours are assigned
            per `color_spots_by`.
        color_spots_by (`'phase'` | `'order'` | `'energy'`): How to colour spots when `spot_color` is `None`.
        i_thresh (float): Minimum intensity as a fraction of the brightest Bragg peak
            (`satellite_order == 0`).  Spots below the cutoff are not
            overlaid.  Default: `0.01` (1 %).  Pass `0.0` to show all spots.
        figsize ((float, float)):
        out_path (str or None): Save figure to this path if provided.

    Returns:
        fig (matplotlib.figure.Figure):
        ax (matplotlib.axes.Axes):
        display_image (ndarray): The image array that was actually plotted — the warped 2θ / χ grid
            when `frame='tth_chi'`, or the (optionally log-scaled) raw pixel
            image when `frame='detector'`.
"""
    if frame not in ("tth_chi", "detector"):
        raise ValueError(f"frame must be 'tth_chi' or 'detector', got {frame!r}")

    # ── Intensity threshold ───────────────────────────────────────────────────
    if spots and i_thresh > 0.0:
        bragg = [s for s in spots if s.get("satellite_order", 0) == 0]
        i_ref = max(s["intensity"] for s in bragg) if bragg else max(s["intensity"] for s in spots)
        spots = [s for s in spots if s["intensity"] >= i_thresh * i_ref]

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor("#000000")

    # ── Image ─────────────────────────────────────────────────────────────────
    if frame == "tth_chi":
        warped, tth_ax, chi_ax = warp_image_to_tth_chi(
            image, camera,
            tth_range=tth_range, chi_range=chi_range,
            n_tth=n_tth, n_chi=n_chi,
        )
        display = np.where(np.isnan(warped), 0.0, warped)
        if log_scale and display.max() > 0:
            display = np.log1p(display / display.max() * 1000.0)
        display[np.isnan(warped)] = np.nan
        extent = [tth_ax[0], tth_ax[-1], chi_ax[0], chi_ax[-1]]
        im = ax.imshow(display, origin="lower", aspect="auto",
                       extent=extent, cmap=cmap, interpolation="nearest")
        ax.set_xlabel("2θ  (°)", color="#7788aa", fontsize=9)
        ax.set_ylabel("χ  (°)", color="#7788aa", fontsize=9)
        frame_label = "2θ / χ frame"
    else:  # 'detector'
        raw = np.asarray(image, dtype=np.float64)
        display = raw.copy()
        if log_scale and display.max() > 0:
            display = np.log1p(display / display.max() * 1000.0)
        # imshow with pixel-coordinate extent; origin='upper' so row 0 is at top
        im = ax.imshow(display, origin="upper", aspect="equal",
                       extent=[0, camera.Nh, camera.Nv, 0],
                       cmap=cmap, interpolation="nearest")
        ax.set_xlabel("x  (px)", color="#7788aa", fontsize=9)
        ax.set_ylabel("y  (px)", color="#7788aa", fontsize=9)
        frame_label = "detector frame"

    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.025)
    cb.set_label("log intensity" if log_scale else "intensity",
                 color=FG, fontsize=8)
    cb.ax.tick_params(colors="#7788aa", labelsize=7)

    # ── Overlay spots ─────────────────────────────────────────────────────────
    lc = {}
    unique = []
    if spots:
        if frame == "tth_chi":
            xs = np.array([s["tth"] for s in spots])
            ys = np.array([s["chi"] for s in spots])
        else:
            # Prefer pre-computed pixel positions; fall back to projection
            xs_list, ys_list = [], []
            for s in spots:
                pix = s.get("pix")
                if pix is not None:
                    xs_list.append(pix[0])
                    ys_list.append(pix[1])
                else:
                    tth_r = np.radians(s["tth"])
                    chi_r = np.radians(s["chi"])
                    st = np.sin(tth_r)
                    kf = np.array([[np.cos(tth_r),
                                    st * np.sin(chi_r),
                                    st * np.cos(chi_r)]])
                    xc, yc = camera.kf_to_pixel(kf)
                    xs_list.append(float(xc[0]))
                    ys_list.append(float(yc[0]))
            xs = np.array(xs_list)
            ys = np.array(ys_list)
            # Drop spots that project off the detector
            on_det = (
                np.isfinite(xs) & np.isfinite(ys)
                & (xs >= 0) & (xs < camera.Nh)
                & (ys >= 0) & (ys < camera.Nv)
            )
            spots = [s for s, ok in zip(spots, on_det) if ok]
            xs = xs[on_det]
            ys = ys[on_det]

        if spot_color is not None:
            colors = spot_color
        elif color_spots_by == "phase":
            labels = [s.get("phase_label", "sim") for s in spots]
            unique = list(dict.fromkeys(labels))
            tab10 = plt.get_cmap("tab10")
            lc = {lb: tab10(i / max(len(unique) - 1, 1)) for i, lb in enumerate(unique)}
            colors = [lc[lb] for lb in labels]
        elif color_spots_by == "order":
            orders = np.array([s.get("satellite_order", 0) for s in spots])
            norm = mcolors.Normalize(vmin=orders.min(), vmax=orders.max())
            colors = plt.get_cmap("coolwarm")(norm(orders))
        elif color_spots_by == "energy":
            energies = np.array([s["E"] / 1e3 for s in spots])
            norm = mcolors.Normalize(vmin=energies.min(), vmax=energies.max())
            colors = plt.get_cmap("plasma")(norm(energies))
        else:
            colors = "#ff4444"

        ax.scatter(xs, ys, c=colors, s=spot_size, marker=spot_marker,
                   linewidths=0.9, zorder=4, label="simulated spots")
        if show_divergence:
            _draw_divergence_ellipses(ax, spots, xs, ys, frame,
                                      divergence_nsigma, colors)

        # Hover tooltip — pass axes coordinates (tth/chi or xcam/ycam)
        _attach_hover_tooltip(fig, ax, spots, xs, ys)

    # ── Axes styling ──────────────────────────────────────────────────────────
    ax.tick_params(colors="#7788aa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1f2e")

    n_spots = len(spots) if spots else 0
    hover_hint = "   (hover for details)" if n_spots else ""
    ax.set_title(
        f"Detector image — {frame_label}"
        + (f"   |   {n_spots} simulated spots{hover_hint}" if n_spots else ""),
        color=FG, fontsize=9, pad=6,
    )

    if spots and spot_color is None and color_spots_by == "phase" and unique:
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], linestyle="none", marker=spot_marker,
                   color=lc[lb], markersize=7, label=lb)
            for lb in unique
        ]
        ax.legend(handles=handles, loc="upper right", fontsize=7.5,
                  framealpha=0.5, facecolor="#1a1f2e", edgecolor="#3a3f4e",
                  labelcolor=FG)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Overlay saved → {out_path}")

    return fig, ax, display


# ─────────────────────────────────────────────────────────────────────────────
# SIDE-BY-SIDE EXPERIMENT / SIMULATION COMPARISON
# ─────────────────────────────────────────────────────────────────────────────

def plot_laue_comparison(
    exp_image,
    sim_image,
    camera,
    spots=None,
    *,
    frame: str = "tth_chi",
    tth_range=None,
    chi_range=None,
    n_tth: int = 600,
    n_chi: int = 600,
    cmap: str = "gray",
    spot_marker: str = "+",
    sat_marker: str = "x",
    spot_size: float = 60.0,
    spot_color=None,
    color_spots_by: str = "phase",
    i_thresh: float = 0.01,
    show_divergence: bool = True,
    divergence_nsigma: float = 2.0,
    figsize=(18, 7),
    out_path: str | None = None,
):
    """
    Side-by-side comparison of an experimental Laue image and a simulated one,
    with simulated spot positions overlaid on the right panel.

    Args:
        exp_image (array-like, shape (Nv, Nh)): Experimental detector image.
        sim_image (array-like, shape (Nv, Nh)): Simulated detector image from :meth:`~Camera.render`.
        camera (Camera): Detector geometry (shared by both panels).
        spots (list[dict], optional): Spot list from any `simulate_laue*` function.  Overlaid on the
            right (simulation) panel only.
        frame (`'tth_chi'` | `'detector'`): Display coordinate frame.

        * `'tth_chi'` — both images are warped to an evenly-spaced 2θ / χ
          grid; spots are placed at their angular coordinates.
        * `'detector'` — raw pixel images; spots projected via their
          `'pix'` key or back-projected from 2θ / χ.
        tth_range, chi_range ((float, float), optional): Angular display range (*tth_chi* frame only).
        n_tth, n_chi (int): Warp resolution (*tth_chi* frame only).
        cmap (str): Matplotlib colormap (applied to both panels).
        spot_marker (str): Marker style for Bragg (satellite_order == 0) spots (default `'+'`).
        sat_marker (str): Marker style for satellite spots (default `'x'`), kept visually
            distinct from *spot_marker* so Bragg and satellite reflections
            don't have to be told apart by size/alpha alone.
        spot_size (float): Marker size.
        spot_color (str or None): Fixed colour for all spots.  `None` → colour by `color_spots_by`.
        color_spots_by (`'phase'` | `'order'` | `'energy'`): Colouring scheme when *spot_color* is `None`.
        i_thresh (float): Minimum `I/Imax` to show a spot (0 = show all).  Spots with
            `intensity < i_thresh` are hidden.  The toggle button shows /
            hides satellite spots on top of this threshold.
        figsize ((float, float)):
        out_path (str or None): Save figure to this path if provided.

    Returns:
        fig (matplotlib.figure.Figure):
        ax_exp (matplotlib.axes.Axes): left (experimental) panel
        ax_sim (matplotlib.axes.Axes): right (simulation) panel
"""
    from matplotlib.widgets import CheckButtons

    if frame not in ("tth_chi", "detector"):
        raise ValueError(f"frame must be 'tth_chi' or 'detector', got {frame!r}")

    # ── Apply intensity threshold ─────────────────────────────────────────────
    spots_all = list(spots) if spots else []
    if spots_all and i_thresh > 0.0:
        i_ref = max(s["intensity"] for s in spots_all)
        spots_all = [s for s in spots_all if s["intensity"] >= i_thresh * i_ref]

    # Split into Bragg and satellite lists
    spots_bragg = [s for s in spots_all if s.get("satellite_order", 0) == 0]
    spots_sat   = [s for s in spots_all if s.get("satellite_order", 0) != 0]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, facecolor=BG)
    # Reserve a thin strip on the right for the checkbox widget
    gs = mgridspec.GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[1, 1, 0.12],
        wspace=0.08,
    )
    ax_exp = fig.add_subplot(gs[0, 0])
    ax_sim = fig.add_subplot(gs[0, 1], sharex=ax_exp, sharey=ax_exp)
    ax_cb  = fig.add_subplot(gs[0, 2])   # checkbox lives here
    ax_cb.set_axis_off()

    for ax in (ax_exp, ax_sim):
        ax.set_facecolor("#000000")
        ax.tick_params(colors="#7788aa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a1f2e")

    # ── Helper: prepare one image for display ─────────────────────────────────
    def _prepare(raw_arr):
        """Return (display_array, extent) ready for imshow."""
        arr = np.asarray(raw_arr, dtype=np.float64)
        if frame == "tth_chi":
            warped, tth_ax, chi_ax = warp_image_to_tth_chi(
                arr, camera,
                tth_range=tth_range, chi_range=chi_range,
                n_tth=n_tth, n_chi=n_chi,
            )
            # Extend by half a pixel so that axis values (pixel centres) align
            # with the markers placed at (tth, chi) coordinates.
            dt = (tth_ax[-1] - tth_ax[0]) / max(len(tth_ax) - 1, 1)
            dc = (chi_ax[-1] - chi_ax[0]) / max(len(chi_ax) - 1, 1)
            ext = [
                tth_ax[0] - dt / 2, tth_ax[-1] + dt / 2,
                chi_ax[0] - dc / 2, chi_ax[-1] + dc / 2,
            ]
            return warped, ext
        else:
            # Use pixel-centre convention: array pixel [r, c] has its centre at
            # data coordinate (c, r).  The default matplotlib imshow extent
            # [-0.5, Nh-0.5, Nv-0.5, -0.5] achieves this.  Without this
            # correction every pixel centre would be shifted +0.5 px in x and y
            # relative to the scatter markers plotted at (xcam, ycam).
            ext = [-0.5, camera.Nh - 0.5, camera.Nv - 0.5, -0.5]
            return arr, ext

    exp_disp, ext = _prepare(exp_image)
    sim_disp, _   = _prepare(sim_image)

    # ── Build LogNorm for simulation panel ───────────────────────────────────
    def _lognorm(arr):
        vals = arr[np.isfinite(arr) & (arr > 0)]
        if vals.size == 0:
            return mcolors.Normalize(vmin=0, vmax=1)
        return mcolors.LogNorm(vmin=vals.min(), vmax=vals.max())

    origin = "lower" if frame == "tth_chi" else "upper"
    aspect = "auto"  if frame == "tth_chi" else "equal"

    # Experimental panel — simple linear log1p scaling
    exp_plot = np.where(np.isfinite(exp_disp) & (exp_disp > 0), exp_disp, np.nan)
    exp_log  = np.log1p(exp_plot)
    im_exp = ax_exp.imshow(
        exp_log, origin=origin, aspect=aspect, extent=ext,
        cmap=cmap, interpolation="nearest",
    )
    cb_exp = fig.colorbar(im_exp, ax=ax_exp, pad=0.02, fraction=0.04)
    cb_exp.set_label("log₁₊(intensity)", color=FG, fontsize=7)
    cb_exp.ax.tick_params(colors="#7788aa", labelsize=6)

    # Simulation panel — LogNorm
    sim_plot = np.where(np.isfinite(sim_disp) & (sim_disp > 0), sim_disp, np.nan)
    sim_norm = _lognorm(sim_plot)
    im_sim = ax_sim.imshow(
        sim_plot, origin=origin, aspect=aspect, extent=ext,
        cmap=cmap, norm=sim_norm, interpolation="nearest",
    )
    cb_sim = fig.colorbar(im_sim, ax=ax_sim, pad=0.02, fraction=0.04)
    cb_sim.set_label("intensity (LogNorm)", color=FG, fontsize=7)
    cb_sim.ax.tick_params(colors="#7788aa", labelsize=6)

    # ── Axis labels ───────────────────────────────────────────────────────────
    if frame == "tth_chi":
        for ax in (ax_exp, ax_sim):
            ax.set_xlabel("2θ  (°)", color="#7788aa", fontsize=9)
        ax_exp.set_ylabel("χ  (°)", color="#7788aa", fontsize=9)
    else:
        for ax in (ax_exp, ax_sim):
            ax.set_xlabel("x  (px)", color="#7788aa", fontsize=9)
        ax_exp.set_ylabel("y  (px)", color="#7788aa", fontsize=9)
    # Hide redundant y tick labels on the shared right panel
    plt.setp(ax_sim.get_yticklabels(), visible=False)

    ax_exp.set_title("Experiment", color=FG, fontsize=9, pad=6)
    ax_sim.set_title("Simulation", color=FG, fontsize=9, pad=6)

    # ── Helper: compute (xs, ys) in display coordinates for a spot list ───────
    def _spot_coords(slist):
        if not slist:
            return np.array([]), np.array([]), slist
        if frame == "tth_chi":
            xs = np.array([s["tth"] for s in slist])
            ys = np.array([s["chi"] for s in slist])
            return xs, ys, slist
        else:
            xs_l, ys_l = [], []
            for s in slist:
                pix = s.get("pix")
                if pix is not None:
                    xs_l.append(pix[0]); ys_l.append(pix[1])
                else:
                    tth_r = np.radians(s["tth"]); chi_r = np.radians(s["chi"])
                    st = np.sin(tth_r)
                    kf = np.array([[np.cos(tth_r), st * np.sin(chi_r), st * np.cos(chi_r)]])
                    xc, yc = camera.kf_to_pixel(kf)
                    xs_l.append(float(xc[0])); ys_l.append(float(yc[0]))
            xs = np.array(xs_l); ys = np.array(ys_l)
            on_det = (
                np.isfinite(xs) & np.isfinite(ys)
                & (xs >= 0) & (xs < camera.Nh)
                & (ys >= 0) & (ys < camera.Nv)
            )
            return xs[on_det], ys[on_det], [s for s, ok in zip(slist, on_det) if ok]

    # ── Colour helper ─────────────────────────────────────────────────────────
    _tab10 = plt.get_cmap("tab10")
    _all_labels = list(dict.fromkeys(
        s.get("phase_label", "sim") for s in spots_all
    ))
    _lc = {lb: _tab10(i / max(len(_all_labels) - 1, 1))
           for i, lb in enumerate(_all_labels)}

    def _colors(slist):
        if spot_color is not None:
            return spot_color
        if color_spots_by == "phase":
            return [_lc[s.get("phase_label", "sim")] for s in slist]
        if color_spots_by == "order":
            ords = np.array([s.get("satellite_order", 0) for s in slist])
            nm = mcolors.Normalize(vmin=ords.min(), vmax=ords.max())
            return plt.get_cmap("coolwarm")(nm(ords))
        if color_spots_by == "energy":
            en = np.array([s["E"] / 1e3 for s in slist])
            nm = mcolors.Normalize(vmin=en.min(), vmax=en.max())
            return plt.get_cmap("plasma")(nm(en))
        return "#ff4444"

    # ── Draw Bragg spots on both panels (always visible) ─────────────────────
    xb, yb, sb = _spot_coords(spots_bragg)
    if sb:
        cb = _colors(sb)
        for _ax in (ax_exp, ax_sim):
            _ax.scatter(
                xb, yb, c=cb,
                s=spot_size, marker=spot_marker, linewidths=1.4, zorder=4,
            )
            if show_divergence:
                _draw_divergence_ellipses(_ax, sb, xb, yb, frame,
                                          divergence_nsigma, cb)
        _attach_hover_tooltip(fig, ax_exp, sb, xb, yb)
        _attach_hover_tooltip(fig, ax_sim, sb, xb, yb)

    # ── Draw satellite spots on both panels (toggleable) ──────────────────────
    xs, ys, ss = _spot_coords(spots_sat)
    sc_sat_exp = sc_sat_sim = None
    if ss:
        cs = _colors(ss)
        sc_sat_exp = ax_exp.scatter(
            xs, ys, c=cs,
            s=spot_size * 0.7, marker=sat_marker, linewidths=1.1,
            zorder=3, alpha=0.75,
        )
        sc_sat_sim = ax_sim.scatter(
            xs, ys, c=cs,
            s=spot_size * 0.7, marker=sat_marker, linewidths=1.1,
            zorder=3, alpha=0.75,
        )
        if show_divergence:
            for _ax in (ax_exp, ax_sim):
                _draw_divergence_ellipses(_ax, ss, xs, ys, frame,
                                          divergence_nsigma, cs)
        _attach_hover_tooltip(fig, ax_exp, ss, xs, ys)
        _attach_hover_tooltip(fig, ax_sim, ss, xs, ys)

    # ── Phase legend ──────────────────────────────────────────────────────────
    phase_legend = None
    if spots_all and spot_color is None and color_spots_by == "phase" and _all_labels:
        handles = [
            Line2D([0], [0], linestyle="none", marker=spot_marker,
                   color=_lc[lb], markersize=7, label=lb)
            for lb in _all_labels
        ]
        phase_legend = ax_sim.legend(
            handles=handles, loc="upper right", fontsize=7,
            framealpha=0.5, facecolor="#1a1f2e", edgecolor="#3a3f4e",
            labelcolor=FG,
        )

    # ── Marker-shape legend (Bragg vs satellite) ─────────────────────────────
    if sb or ss:
        shape_handles = []
        if sb:
            shape_handles.append(Line2D([0], [0], linestyle="none", marker=spot_marker,
                                         color="#cccccc", markersize=7, label="Bragg"))
        if ss:
            shape_handles.append(Line2D([0], [0], linestyle="none", marker=sat_marker,
                                         color="#cccccc", markersize=7, label="satellite"))
        ax_sim.legend(
            handles=shape_handles, loc="lower right", fontsize=7,
            framealpha=0.5, facecolor="#1a1f2e", edgecolor="#3a3f4e",
            labelcolor=FG,
        )
        if phase_legend is not None:
            ax_sim.add_artist(phase_legend)

    # ── Satellite toggle checkbox ─────────────────────────────────────────────
    # Store the widget on the figure so it is not garbage-collected.
    chk_ax = fig.add_axes([0.91, 0.46, 0.08, 0.08], facecolor="#1a1f2e")
    fig._sat_chk = CheckButtons(chk_ax, ["satellites"], [sc_sat_sim is not None])
    fig._sat_chk.labels[0].set_color(FG)
    fig._sat_chk.labels[0].set_fontsize(8)

    def _toggle_sat(_):
        for sc in (sc_sat_exp, sc_sat_sim):
            if sc is not None:
                sc.set_visible(not sc.get_visible())
        fig.canvas.draw_idle()

    fig._sat_chk.on_clicked(_toggle_sat)

    n_spots = len(spots_all)
    ax_sim.set_title(
        f"Simulation  |  {n_spots} spots  (hover for details)",
        color=FG, fontsize=9, pad=6,
    )

    fig.tight_layout(rect=[0, 0, 0.90, 1.0])
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Comparison saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# HKL-FAMILY SURROUNDINGS — INTERACTIVE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────


def plot_hkl_family_classification(
    image,
    peaklist,
    spots,
    hkl,
    *,
    n_multiples: int = 5,
    i_thresh: float = 0.0,
    crop_half_size: int = 20,
    n_cols: int = 5,
    color_by: str = "order",
    spot_size: float = 90.0,
    spot_alpha: float = 1.0,
    cmap: str = "inferno",
    figsize: tuple | None = None,
    out_path: str | None = None,
):
    """
    Zoom into the surroundings of a lattice-plane (hkl) family and let you
    manually pick, per zoomed panel, which candidate simulated spot is the
    real origin of the measured peak there.

    Why this exists: a systematic row of satellite reflections (m=0 Bragg
    peak plus m=±1, ±2, … satellites) often lands so close together on the
    detector that several candidates fall inside the same few pixels, and
    it isn't obvious by eye — or by simple nearest-neighbour matching —
    which one a given measured peak actually corresponds to. This function
    crops the raw image around each such cluster so you can zoom in and
    decide by hand.

    Simulated spots matching `hkl = m * (h, k, l)` (for `m = ±1 …
    ±n_multiples`, see :func:`print_hkl_family`) are grouped into panels:
    spots that land close enough together on the detector to fit in one
    crop window (typically a Bragg reflection together with its satellite
    orders) share a single panel. Each panel shows two *different* things
    overlaid on the same crop — don't confuse them:

    * coloured markers — **predictions**: where the simulation says each
      candidate spot should be. One marker per candidate in the group,
      colour-coded by *color_by* and shaped by which phase it originates
      from (hover a marker for its full details — hkl, energy, intensity,
      satellite order, phase).
    * hollow white circles — **observations**: real peaks found by the
      segmentation pipeline (from *peaklist*) that fall inside the crop.
      There may be zero, one, or several.

    Example: a panel shows one white circle (a real peak) sitting right on
    top of the orange `'+'` (predicted m=+1 satellite) but 4 px away from
    the blue `'+'` (predicted m=0 Bragg spot). Clicking the orange marker
    records "this measured peak is the m=+1 satellite, not the Bragg
    spot" as that panel's classification.

    Mechanically, clicking is just picking one of the candidate markers as
    your answer for that panel — it does not move, fit, or alter anything,
    it only records a label:

    * click near a candidate marker → that candidate is stored as the
      panel's classification; the panel border is recoloured to match it
      and the title shows the chosen `(hkl)` / satellite order.
    * click on empty space inside a panel (not near any marker) → clears
      that panel's classification.

    Classifications are visual bookkeeping only, kept in memory as
    `fig._classifications` (dict: panel index → the classified spot's
    dict) for as long as the figure stays open — there is no auto-save;
    read that attribute yourself if you want to keep the picks.

    Args:
        image ((Nv, Nh) array): Raw detector image, in the same pixel frame as *peaklist* and the
            `'pix'` key of *spots*.
        peaklist (ndarray (N, >=2), or DataFrame): Segmented peak table. Columns 0, 1 (or `'peak_X'`, `'peak_Y'`)
            are pixel positions.
        spots (list[dict]): Spot list from any `simulate_laue*` function. Each dict must contain
            `'hkl'` and `'pix'`.
        hkl ((int, int, int)): Base Miller indices `(h, k, l)` of the lattice-plane family to
            inspect.
        n_multiples (int): Highest multiple of *hkl* to search for. Default 5.
        i_thresh (float): Minimum `intensity / max(intensity)`, evaluated over the matched
            family only, for a candidate to be shown. Default `0.0` (show
            every match).
        crop_half_size (int): Half-width, in pixels, of each crop window. Also the clustering
            radius: simulated spots within `2 * crop_half_size` of each
            other share one panel.
        n_cols (int): Number of panels per row.
        color_by (`'order'` | `'phase'`): Candidate *colour* coding. `'order'` — satellite order `m`
            (coolwarm, `m=0` = Bragg peak). `'phase'` — `phase_label`
            (tab10). Independently of *color_by*, candidates are also given
            a marker *shape* per `phase_label` (e.g. all `'bcc'` candidates
            drawn as `'+'`, all `'fcc'` as `'x'`, a third phase as `'^'`,
            …) so the phase a diffraction event originates from stays
            visible even when *color_by* is `'order'`. `'o'` is never used
            for a candidate — it is reserved for the measured-peak
            circles. A marker-shape legend is added automatically when
            more than one phase is present.
        spot_size (float): Marker area (points²) for the candidate simulated-spot markers.
            Default `90.0`.
        spot_alpha (float): Opacity of the candidate simulated-spot markers, in `[0, 1]`.
            Default `1.0` (opaque). Lower it to make an occluding candidate
            marker translucent so a measured-peak circle underneath it, or
            an overlapping candidate, stays visible.
        cmap (str): Colormap for the background image crops.
        figsize ((float, float), optional): Figure size in inches. Defaults to a size scaled to
            the number of panels.
        out_path (str or None): Save a static PNG snapshot (before any classification clicks)
            if provided.

    Returns:
        fig (matplotlib.figure.Figure):
        axes (list[matplotlib.axes.Axes]): One axes per panel, in display order.
"""
    if color_by not in ("order", "phase"):
        raise ValueError(f"color_by must be 'order' or 'phase', got {color_by!r}")

    # ── select the (h,k,l) family ──────────────────────────────────────────────
    h0, k0, l0 = (int(v) for v in hkl)
    targets = set()
    for m in range(1, int(n_multiples) + 1):
        targets.add((m * h0, m * k0, m * l0))
        targets.add((-m * h0, -m * k0, -m * l0))

    matches = [
        s for s in spots
        if s.get("pix") is not None
        and tuple(int(x) for x in s["hkl"]) in targets
    ]
    if not matches:
        raise ValueError(
            f"No spots with a 'pix' position found for hkl family {hkl} "
            f"(multiples 1..{n_multiples})."
        )
    if i_thresh > 0.0:
        i_ref = max(s["intensity"] for s in matches)
        matches = [s for s in matches if s["intensity"] >= i_thresh * i_ref]

    # ── group nearby candidates so they share one crop window ─────────────────
    xy = np.array([s["pix"] for s in matches], dtype=float)
    tree = KDTree(xy)
    parent = list(range(len(matches)))

    def _find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in tree.query_pairs(r=2.0 * crop_half_size):
        ri, rj = _find(i), _find(j)
        if ri != rj:
            parent[ri] = rj

    groups = {}
    for i in range(len(matches)):
        groups.setdefault(_find(i), []).append(matches[i])

    sites = []
    for cand in groups.values():
        cx = float(np.mean([c["pix"][0] for c in cand]))
        cy = float(np.mean([c["pix"][1] for c in cand]))
        sites.append({"candidates": cand, "center": (cx, cy)})
    sites.sort(key=lambda site: site["center"][1])

    # ── parse the measured peaklist ─────────────────────────────────────────
    try:
        import pandas as pd
        if isinstance(peaklist, pd.DataFrame):
            pk_xy = peaklist[["peak_X", "peak_Y"]].to_numpy(dtype=float)
        else:
            pk_xy = np.asarray(peaklist, dtype=float)[:, :2]
    except ImportError:
        pk_xy = np.asarray(peaklist, dtype=float)[:, :2]

    # ── candidate colour coding (shared across all panels) ─────────────────
    labels = list(dict.fromkeys(s.get("phase_label", "sim") for s in matches))
    if color_by == "order":
        orders = np.array([s.get("satellite_order", 0) for s in matches])
        omax = max(abs(int(orders.min())), abs(int(orders.max())), 1)
        norm = mcolors.Normalize(vmin=-omax, vmax=omax)
        order_cmap = plt.get_cmap("coolwarm")
        color_of = lambda s: order_cmap(norm(s.get("satellite_order", 0)))
    else:
        tab10 = plt.get_cmap("tab10")
        lc = {lb: tab10(i / max(len(labels) - 1, 1)) for i, lb in enumerate(labels)}
        color_of = lambda s: lc[s.get("phase_label", "sim")]

    # ── candidate marker shape, one per phase (independent of colour) ──────
    # "o" is reserved for measured-peak circles, so it is never used here.
    _CANDIDATE_MARKERS = ["+", "x", "^", "s", "D", "v", "p", "h", "P", "*"]
    marker_of_phase = {
        lb: _CANDIDATE_MARKERS[i % len(_CANDIDATE_MARKERS)]
        for i, lb in enumerate(labels)
    }
    marker_of = lambda s: marker_of_phase[s.get("phase_label", "sim")]

    # ── figure / axes grid ───────────────────────────────────────────────────
    n_sites = len(sites)
    n_cols = max(1, min(n_cols, n_sites))
    n_rows = int(np.ceil(n_sites / n_cols))
    if figsize is None:
        figsize = (2.6 * n_cols, 2.6 * n_rows + 0.7)

    fig, axes_grid = plt.subplots(n_rows, n_cols, figsize=figsize, facecolor=BG)
    axes = np.atleast_1d(axes_grid).ravel().tolist()
    for ax in axes[n_sites:]:
        ax.set_visible(False)
    axes = axes[:n_sites]

    img = np.asarray(image, dtype=float)
    Nv, Nh = img.shape

    classifications = {}
    UNSET_EDGE = "#1a1f2e"

    def _panel_title(i, n_cand):
        return f"panel {i}  ·  {n_cand} candidate(s)"

    for i, (ax, site) in enumerate(zip(axes, sites)):
        cx, cy = site["center"]
        r0 = int(round(cy - crop_half_size))
        r1 = int(round(cy + crop_half_size))
        c0 = int(round(cx - crop_half_size))
        c1 = int(round(cx + crop_half_size))
        r0c, r1c = max(r0, 0), min(r1, Nv)
        c0c, c1c = max(c0, 0), min(c1, Nh)
        crop = img[r0c:r1c, c0c:c1c]

        ax.set_facecolor("#000000")
        ax.tick_params(colors="#7788aa", labelsize=6)
        for sp in ax.spines.values():
            sp.set_edgecolor(UNSET_EDGE)
            sp.set_linewidth(1.5)

        disp = np.where(np.isfinite(crop) & (crop > 0), crop, np.nan)
        ax.imshow(
            np.log1p(disp), origin="upper", cmap=cmap,
            extent=[c0c - 0.5, c1c - 0.5, r1c - 0.5, r0c - 0.5],
            interpolation="nearest", aspect="equal",
        )

        # measured peaks inside this crop
        if pk_xy.shape[0] > 0:
            in_crop = (
                (pk_xy[:, 0] >= c0c) & (pk_xy[:, 0] < c1c)
                & (pk_xy[:, 1] >= r0c) & (pk_xy[:, 1] < r1c)
            )
            if in_crop.any():
                ax.scatter(
                    pk_xy[in_crop, 0], pk_xy[in_crop, 1],
                    s=70, marker="o", facecolors="none",
                    edgecolors="white", linewidths=1.0, alpha=0.85, zorder=3,
                )

        cand = site["candidates"]
        xs = np.array([c["pix"][0] for c in cand])
        ys = np.array([c["pix"][1] for c in cand])
        cols = [color_of(c) for c in cand]
        mks = [marker_of(c) for c in cand]
        # scatter() takes one marker per call, so candidates with different
        # phases (=> different marker shapes) are drawn in separate calls.
        for mk in set(mks):
            sel = [j for j, m in enumerate(mks) if m == mk]
            is_line_marker = mk in ("+", "x")
            ax.scatter(
                xs[sel], ys[sel], c=[cols[j] for j in sel],
                marker=mk, s=spot_size, alpha=spot_alpha, linewidths=1.4,
                edgecolors=None if is_line_marker else "white",
                zorder=4,
            )
        _attach_hover_tooltip(fig, ax, cand, xs, ys)

        ax.set_xlim(c0c - 0.5, c1c - 0.5)
        ax.set_ylim(r1c - 0.5, r0c - 0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(_panel_title(i, len(cand)), color=FG, fontsize=7, pad=3)

        ax._family_candidates = cand
        ax._family_colors = cols
        ax._family_xy = np.column_stack([xs, ys])
        ax._family_index = i

    # ── click-to-classify ────────────────────────────────────────────────────
    def _on_click(event):
        ax = event.inaxes
        if ax is None or not hasattr(ax, "_family_candidates"):
            return

        cand = ax._family_candidates
        xy_disp = ax.transData.transform(ax._family_xy)
        mouse_disp = np.array([event.x, event.y])
        dists = np.linalg.norm(xy_disp - mouse_disp, axis=1)
        threshold_px = 10.0 * fig.dpi / 72.0
        idx = int(np.argmin(dists))
        i = ax._family_index

        if dists[idx] <= threshold_px:
            chosen = cand[idx]
            classifications[i] = chosen
            hh, kk, ll = chosen["hkl"]
            order = chosen.get("satellite_order", 0)
            order_txt = f"  m={order:+d}" if order else ""
            phase_txt = f"  [{chosen.get('phase_label')}]" if len(labels) > 1 else ""
            ax.set_title(f"({hh:+d}{kk:+d}{ll:+d}){order_txt}{phase_txt}  ✓",
                         color=FG, fontsize=7, pad=3)
            edge_col = ax._family_colors[idx]
            lw = 2.5
        else:
            classifications.pop(i, None)
            ax.set_title(_panel_title(i, len(cand)), color=FG, fontsize=7, pad=3)
            edge_col = UNSET_EDGE
            lw = 1.5

        for sp in ax.spines.values():
            sp.set_edgecolor(edge_col)
            sp.set_linewidth(lw)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", _on_click)
    fig._classifications = classifications

    fig.suptitle(
        f"({h0:+d}{k0:+d}{l0:+d}) family  —  {n_sites} panel(s), "
        f"{len(matches)} candidate spot(s)  |  click a marker to classify, "
        f"click empty space to clear",
        color=FG, fontsize=9, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 0.93, 0.96])

    # ── shared colour legend (added after layout so it gets its own space) ────
    if color_by == "order":
        cax = fig.add_axes([0.945, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap=order_cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label("satellite order  m", color=FG, fontsize=8)
        cb.ax.tick_params(colors="#7788aa", labelsize=7)
        cb.outline.set_edgecolor("#1a1f2e")
        phase_legend_color = FG   # colour already used for order; keep markers neutral
    else:
        phase_legend_color = None  # use each phase's own colour from lc

    # Marker-shape legend — only meaningful when more than one phase is present.
    if len(labels) > 1:
        handles = [
            Line2D([0], [0], linestyle="none", marker=marker_of_phase[lb],
                   color=lc[lb] if phase_legend_color is None else phase_legend_color,
                   markersize=8, label=lb)
            for lb in labels
        ]
        fig.legend(
            handles=handles, loc="upper right", fontsize=7,
            framealpha=0.5, facecolor="#1a1f2e", edgecolor="#3a3f4e",
            labelcolor=FG, title="phase (marker shape)",
            title_fontsize=7,
        )

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  HKL-family classification plot saved -> {out_path}")

    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# SEGMENTATION QUALITY
# ─────────────────────────────────────────────────────────────────────────────


def plot_segmentation(
    image: "np.ndarray",
    peaklist: "np.ndarray",
    *,
    log_scale: bool = True,
    color_by_intensity: bool = True,
    show_ellipses: bool = False,
    marker_size: float = 60.0,
    cmap_image: str = "inferno",
    cmap_markers: str = "plasma",
    vmin_pct: float = 1.0,
    vmax_pct: float = 99.5,
    figsize: tuple = (10, 8),
    title: str | None = None,
    ax=None,
):
    """
    Display a detector image with segmented peak positions overlaid as "+"
    symbols.

    Typical usage::

        image    = load_images("scan.h5")[0]
        peaklist = convert_spotsfile2peaklist("scan_spots.h5")
        fig, ax  = plot_segmentation(image, peaklist)

    Args:
        image ((Nv, Nh) array): Raw detector frame.
        peaklist ((N, ≥2) array): Peak table from :func:`~nrxrdct.laue.convert_spotsfile2peaklist`.
            Column layout (from that function):

        * 0 – `peak_X`  (xcam, column pixel)
        * 1 – `peak_Y`  (ycam, row pixel)
        * 2 – `peak_I`  (fitted peak intensity, optional)
        * 3 – `peak_fwaxmaj`   \\
        * 4 – `peak_fwaxmin`    > required for *show_ellipses*
        * 5 – `peak_inclination` (degrees) /

        A plain (N, 2) array of pixel positions also works.
        log_scale (bool): Apply `log1p` compression to the image before display (default
            `True`).
        color_by_intensity (bool): Colour "+" markers by peak intensity (log-scaled, column 2).
            Falls back to a fixed green if column 2 is absent.
        show_ellipses (bool): Overlay the fitted peak ellipses (columns 3-5 required).
        marker_size (float): Marker area in points².
        cmap_image (str): Colormap for the image (default `"inferno"`).
        cmap_markers (str): Colormap for intensity-coded markers (default `"plasma"`).
        vmin_pct, vmax_pct (float): Percentile clip for the image display range.
        figsize ((w, h)): Figure size in inches (ignored when *ax* is supplied).
        title (str or None): Axes title.  Defaults to `"Segmentation — N spots found"`.
        ax (matplotlib.axes.Axes or None): Draw into an existing axes rather than creating a new figure.

    Returns:
        fig (matplotlib.figure.Figure):
        ax (matplotlib.axes.Axes):
"""
    pl  = np.asarray(peaklist, dtype=float)
    img = np.asarray(image, dtype=float)
    n_spots = len(pl)
    xcam = pl[:, 0]
    ycam = pl[:, 1]

    # ── axes setup ────────────────────────────────────────────────────────────
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors="#7788aa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1f2e")

    # ── image display ─────────────────────────────────────────────────────────
    positive = img[img > 0]
    if positive.size:
        v0 = np.percentile(positive, vmin_pct)
        v1 = np.percentile(positive, vmax_pct)
    else:
        v0, v1 = 0.0, 1.0

    if log_scale:
        denom = max(v1, 1e-10)
        disp = np.log1p(np.clip(img, 0.0, None) / denom * 1000.0)
    else:
        disp = np.clip(img, v0, v1)

    ax.imshow(disp, origin="upper", cmap=cmap_image, aspect="equal", zorder=0)

    # ── spot markers ──────────────────────────────────────────────────────────
    has_intensity = pl.shape[1] >= 3

    if color_by_intensity and has_intensity:
        log_I = np.log1p(np.clip(pl[:, 2], 0.0, None))
        norm  = mcolors.Normalize(
            vmin=np.percentile(log_I, 5) if len(log_I) > 1 else 0.0,
            vmax=np.percentile(log_I, 99) if len(log_I) > 1 else 1.0,
        )
        sc = ax.scatter(
            xcam, ycam,
            s=marker_size, c=log_I, cmap=cmap_markers, norm=norm,
            marker="+", linewidths=1.2, zorder=3,
        )
        cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01, shrink=0.85)
        cb.set_label("log(I + 1)", color=FG, fontsize=8)
        cb.ax.tick_params(colors="#7788aa", labelsize=7)
    else:
        ax.scatter(
            xcam, ycam,
            s=marker_size, c="#00ff88",
            marker="+", linewidths=1.2, zorder=3,
        )

    # ── optional fitted ellipses ──────────────────────────────────────────────
    if show_ellipses and pl.shape[1] >= 6:
        from matplotlib.patches import Ellipse

        fwmaj  = pl[:, 3]
        fwmin  = pl[:, 4]
        angles = pl[:, 5]
        for x, y, a, b, theta in zip(xcam, ycam, fwmaj, fwmin, angles):
            ell = Ellipse(
                (x, y), width=float(a), height=float(b), angle=float(theta),
                fill=False, edgecolor="#44dd66",
                linewidth=0.6, alpha=0.7, zorder=2,
            )
            ax.add_patch(ell)

    # ── labels ────────────────────────────────────────────────────────────────
    ax.set_xlabel("xcam  (px)", color=FG, fontsize=8)
    ax.set_ylabel("ycam  (px)", color=FG, fontsize=8)

    _title = title if title is not None else f"Segmentation — {n_spots} spots found"
    ax.set_title(_title, color=FG, fontsize=9, pad=5)

    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-GRAIN OVERLAY
# ─────────────────────────────────────────────────────────────────────────────

_GRAIN_COLORS = [
    "#4fc3f7",  # sky blue
    "#ff6633",  # orange
    "#66ff99",  # mint
    "#ffcc00",  # yellow
    "#cc88ff",  # violet
    "#ff88bb",  # pink
    "#44dddd",  # teal
]


def plot_multigrain(
    obs_xy: "np.ndarray",
    spots_per_grain: "list[list[dict]]",
    camera,
    *,
    image: "np.ndarray | None" = None,
    match_px: float = 10.0,
    color_obs_by_grain: bool = True,
    figsize: tuple = (9, 8),
    out_path: "str | None" = None,
):
    """
    Overlay observed spots and per-grain simulations on the detector plane.

    Each grain's simulated spots are drawn in a distinct colour; thin lines
    connect each simulated spot to its nearest observed counterpart (within
    *match_px*).  When *color_obs_by_grain* is `True` the observed spots are
    also coloured by whichever grain's simulation is closest to them, giving an
    immediate visual assignment map.

    Typical usage::

        # simulate each grain after fit_orientation_mixed
        spots_per_grain = [
            laue.simulate_laue(crystal, U, camera, f2_thresh=0.01)
            for U in result.U_phases
        ]
        fig, ax = plot_multigrain(peaks[:, :2], spots_per_grain, camera)

    Args:
        obs_xy ((N, 2) array-like): Observed pixel positions `[xcam, ycam]`.
        spots_per_grain (list of spot-lists): One spot list per grain, each in the format returned by
            :func:`~nrxrdct.laue.simulation.simulate_laue`.
            Each spot dict must contain a `'pix'` key `(xcam, ycam)`.
        camera (Camera): Detector geometry; `camera.Nh` and `camera.Nv` set the axis limits.
        image ((Nv, Nh) array or None): Optional raw detector image displayed as a log-scaled background.
        match_px (float): Maximum pixel distance for drawing a match line between a simulated
            spot and its nearest observed spot.  Default `10.0`.
        color_obs_by_grain (bool): When `True` (default), repaint each observed spot in the colour of
            its closest grain.  Unmatched spots (all grains farther than
            *match_px*) are shown in white.
        figsize ((float, float)): Figure size in inches.
        out_path (str or None): If given, save the figure to this path at 150 dpi.

    Returns:
        fig (matplotlib.figure.Figure):
        ax (matplotlib.axes.Axes):
"""
    obs_xy = np.asarray(obs_xy, dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    # ── optional background image ─────────────────────────────────────────────
    if image is not None:
        img = np.asarray(image, dtype=float)
        vmin = np.percentile(img, 1)
        ax.imshow(
            np.log1p(np.clip(img - vmin, 0, None)),
            origin="upper",
            cmap="inferno",
            extent=[0, camera.Nh, camera.Nv, 0],
            aspect="auto",
            interpolation="nearest",
            alpha=0.6,
            zorder=0,
        )

    # ── collect simulated xy arrays per grain ─────────────────────────────────
    sim_xys = []
    for spots in spots_per_grain:
        xy = np.array(
            [s["pix"] for s in spots if s.get("pix") is not None],
            dtype=float,
        )
        sim_xys.append(xy if len(xy) else np.empty((0, 2), dtype=float))

    # ── assign each observed spot to its closest grain ────────────────────────
    obs_grain = np.full(len(obs_xy), -1, dtype=int)   # -1 = unmatched
    if color_obs_by_grain and sim_xys:
        best_dist = np.full(len(obs_xy), np.inf)
        for gi, sxy in enumerate(sim_xys):
            if len(sxy) == 0:
                continue
            diff = obs_xy[:, None, :] - sxy[None, :, :]           # (N_obs, N_sim, 2)
            d = np.sqrt((diff ** 2).sum(axis=-1)).min(axis=1)      # (N_obs,)
            closer = d < best_dist
            best_dist = np.where(closer, d, best_dist)
            obs_grain = np.where(closer, gi, obs_grain)
        obs_grain[best_dist > match_px] = -1

    # ── draw observed spots ───────────────────────────────────────────────────
    unmatched = obs_grain == -1
    if unmatched.any():
        ax.scatter(
            obs_xy[unmatched, 0], obs_xy[unmatched, 1],
            s=40, facecolors="none", edgecolors="white", lw=0.8, zorder=2,
        )
    if color_obs_by_grain:
        for gi in range(len(spots_per_grain)):
            mask = obs_grain == gi
            if mask.any():
                ax.scatter(
                    obs_xy[mask, 0], obs_xy[mask, 1],
                    s=40, facecolors="none",
                    edgecolors=_GRAIN_COLORS[gi % len(_GRAIN_COLORS)],
                    lw=1.2, zorder=2,
                )

    # ── draw simulated spots + match lines per grain ──────────────────────────
    legend_handles = [
        Line2D(
            [0], [0], linestyle="none", marker="o", markersize=6,
            markerfacecolor="none", markeredgecolor="white", lw=0.8,
            label="observed",
        ),
    ]

    for gi, (spots, sxy) in enumerate(zip(spots_per_grain, sim_xys)):
        color = _GRAIN_COLORS[gi % len(_GRAIN_COLORS)]
        n_sim = len(sxy)
        if n_sim == 0:
            continue

        ax.scatter(sxy[:, 0], sxy[:, 1], s=80, marker="+",
                   color=color, lw=1.2, zorder=3)

        nn_dist = np.full(n_sim, np.inf)
        if len(obs_xy):
            diff = sxy[:, None, :] - obs_xy[None, :, :]           # (N_sim, N_obs, 2)
            dist = np.sqrt((diff ** 2).sum(axis=-1))               # (N_sim, N_obs)
            nn_idx = dist.argmin(axis=1)
            nn_dist = dist[np.arange(n_sim), nn_idx]
            for j in range(n_sim):
                if nn_dist[j] < match_px:
                    ox, oy = obs_xy[nn_idx[j]]
                    ax.plot([sxy[j, 0], ox], [sxy[j, 1], oy],
                            color=color, lw=0.5, alpha=0.4, zorder=1)

        n_matched = int((nn_dist < match_px).sum())
        rate = n_matched / max(n_sim, 1)
        legend_handles.append(
            Line2D(
                [0], [0], linestyle="none", marker="+", markersize=8,
                color=color, lw=1.2,
                label=f"grain {gi + 1}  ({n_matched}/{n_sim}, {rate:.0%})",
            )
        )

    # ── axes styling ──────────────────────────────────────────────────────────
    ax.set_xlim(0, camera.Nh)
    ax.set_ylim(camera.Nv, 0)
    ax.set_xlabel("xcam  (px)", color=FG, fontsize=8)
    ax.set_ylabel("ycam  (px)", color=FG, fontsize=8)
    ax.tick_params(colors="#7788aa", labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1f2e")
    ax.set_title(
        f"Multi-grain Laue  —  {len(spots_per_grain)} grain(s)",
        color=FG, fontsize=9, pad=5,
    )
    ax.legend(
        handles=legend_handles,
        facecolor="#1a1f2e", edgecolor="#3a3f4e",
        labelcolor=FG, fontsize=8, loc="upper right",
    )

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# DEPTH-ELONGATION PLOT
# ─────────────────────────────────────────────────────────────────────────────

def _kf_hat_from_spot(spot):
    """Reconstruct unit scattered-beam vector from tth / chi stored in a spot dict."""
    tth = np.radians(float(spot["tth"]))
    chi = np.radians(float(spot["chi"]))
    return np.array([
        np.cos(tth),
        np.sin(tth) * np.sin(chi),
        np.sin(tth) * np.cos(chi),
    ])


def _stack_interface_depths_mm(stack) -> list:
    """
    Return ``[(z_mm, label), ...]`` for each interface to be drawn as a
    vertical line on a depth-axis plot, ordered surface (z=0) to deepest.

    * Each repeating block (``stack.blocks``, surface-most first): if it
      repeats more than once, only its outer boundaries are returned (start
      and end of the whole block) with internal sub-layer boundaries
      suppressed; a block with `n_rep == 1` (e.g. a non-repeating cap layer)
      shows every sub-layer interface instead.
    * Buffer layers (``stack.buffer_layers``): every interface is returned.
    """
    interfaces = []
    z_A = 0.0
    for blk in reversed(stack.blocks):
        period_A = sum(l.thickness for l in blk.layers) if blk.layers else 0.0
        if period_A <= 0:
            continue
        if blk.n_rep > 1:
            interfaces.append((z_A * 1e-7, f"×{blk.n_rep} start"))
            z_A += period_A * blk.n_rep
            interfaces.append((z_A * 1e-7, f"×{blk.n_rep} end"))
        else:
            for layer in reversed(blk.layers):
                z_A += layer.thickness
                interfaces.append((z_A * 1e-7, getattr(layer, "label", "") or ""))

    for layer in reversed(stack.buffer_layers):
        z_A += layer.thickness
        interfaces.append((z_A * 1e-7, getattr(layer, "label", "") or "buffer"))

    return interfaces


def _surface_to_depth_segments(stack):
    """
    Return a list of (z_start_Å, z_end_Å, layer) tuples ordered from the
    crystal surface down to the deepest buffer layer.

    Every repeating block is unrolled (n_rep copies, top repetition first),
    surface-most block first; buffer layers follow in shallowest-first order.
    """
    segments = []
    z = 0.0
    for blk in reversed(stack.blocks):
        for _ in range(blk.n_rep - 1, -1, -1):
            for layer in reversed(blk.layers):
                segments.append((z, z + layer.thickness, layer))
                z += layer.thickness
    for layer in reversed(stack.buffer_layers):
        segments.append((z, z + layer.thickness, layer))
        z += layer.thickness
    return segments


def plot_depth_elongation(
    spots: list[dict],
    stack: LayeredCrystal,
    camera: Camera,
    ki_hat: "np.ndarray | None" = None,
    *,
    top_n: int = 15,
    min_intensity: float = 0.02,
    n_steps_per_layer: int = 8,
    space: str = "detector",
    show_divergence: bool = True,
    divergence_nsigma: float = 2.0,
    image: "np.ndarray | None" = None,
    figsize: tuple[float, float] = (10, 8),
    ax: "plt.Axes | None" = None,
    out_path: "str | None" = None,
):
    """
    Visualise depth-parallax spot elongation for each Laue spot.

    For every spot three graphical elements are drawn:

    * **Trail** — a sequence of projected detector positions stepping from
      the top of the contributing layer to its bottom (or until Beer-Lambert
      absorption reduces transmission below 1e-4).  Line opacity fades with
      depth to reflect the absorption weight.  In ``'angles'`` space a
      single point is drawn instead (depth does not shift 2θ/χ).
    * **Tick markers** (``|``) — low-opacity ticks at the trail endpoints
      (layer top and bottom).
    * **Circle** — a filled circle at ``spot['pix']``, i.e. the position
      the simulation placed the spot.  When the simulation was run with
      ``correct_depth=True`` this is the layer-centre depth position; when
      run without it is the surface position (z = 0).

    When ``show_divergence=True`` a combined broadening ellipse is drawn,
    centred at the absorption-weighted mean of the trail.  Its covariance is
    the sum of the depth-parallax covariance (computed from the trail
    positions weighted by Beer-Lambert absorption) and the beam-divergence
    covariance stored in ``spot['cov_px']`` (populated by
    :func:`beam_divergence_ellipses` when ``sigma_h/v_mrad`` are passed to
    the simulation).  The ellipse major axis therefore automatically aligns
    with the dominant elongation direction — depth-parallax for thick
    layers, beam divergence for thin layers.

    Args:
        spots: Output of :func:`simulate_laue_stack` or
            :func:`simulate_laue_darwin`.  Required keys: ``'tth'``,
            ``'chi'``, ``'E'``, ``'pix'``, ``'intensity'``, ``'phase_label'``.
            Optional: ``'cov_px'`` (added by :func:`beam_divergence_ellipses`).
        stack: The same :class:`LayeredCrystal` used to produce *spots*.
            Provides layer thicknesses, absorption coefficients, and the
            surface-normal direction.
        camera: Detector geometry used in the simulation.
        ki_hat: Incident-beam direction in the LT frame (3-vector).
            Defaults to ``[1, 0, 0]``.
        top_n: Maximum number of spots to plot (strongest first).
        min_intensity: Skip spots below this normalised intensity.
        n_steps_per_layer: Depth samples taken per layer.  More steps give
            smoother trails at the cost of computation time.
        space: ``'detector'`` → pixel (col, row); ``'angles'`` → (2θ °, χ °).
        show_divergence: When ``True`` (default) draw the combined
            depth-parallax + beam-divergence broadening ellipse for each
            spot.  Ellipses are centred at the absorption-weighted mean of
            the trail, not at the simulation dot.
        divergence_nsigma: Scale factor for the broadening ellipse in units
            of σ.  Default 2.0 (≈ 86 % enclosed probability in 2-D).
        image: Optional detector image shown as a greyscale background
            (only used when ``space='detector'``).
        figsize: Figure size in inches.
        ax: Draw into an existing :class:`~matplotlib.axes.Axes`; ``None``
            creates a new figure.
        out_path: Save path; ``None`` → do not save.

    Returns:
        ``(fig, ax)``
    """
    from matplotlib.collections import LineCollection
    from matplotlib.cm import get_cmap

    ki = np.asarray(ki_hat if ki_hat is not None else [1.0, 0.0, 0.0], dtype=float)
    ki /= np.linalg.norm(ki)

    # cos(angle between surface normal and beam) — used for depth→beam conversion
    cos_in = max(abs(float(np.dot(ki, stack.n_hat))), 1e-3)

    # Layer sequence from surface to deep
    segments = _surface_to_depth_segments(stack)
    total_thickness_mm = sum(seg[1] - seg[0] for seg in segments) * 1e-7  # Å→mm

    # ── Filter spots ──────────────────────────────────────────────────────────
    candidates = [s for s in spots
                  if s.get("pix") is not None
                  and float(s.get("intensity", 1.0)) >= min_intensity]
    candidates.sort(key=lambda s: s.get("intensity", 0.0), reverse=True)
    candidates = candidates[:top_n]

    if not candidates:
        raise ValueError("No spots survive the intensity / top_n filter.")

    # ── Colour palette — one colour per phase label ───────────────────────────
    phase_labels = list(dict.fromkeys(
        s.get("phase_label", "unknown") for s in candidates
    ))
    palette = get_cmap("tab10")
    phase_color = {ph: palette(i % 10) for i, ph in enumerate(phase_labels)}

    # ── Set up axes ───────────────────────────────────────────────────────────
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(BG)
    else:
        fig = ax.figure

    ax.set_facecolor(BG)
    ax.tick_params(colors=FG, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1f2e")

    if space == "detector" and image is not None:
        vmax = np.percentile(image, 99.9)
        ax.imshow(image, cmap="gray", vmin=0, vmax=vmax,
                  aspect="equal", interpolation="nearest")

    # ── Per-spot depth trails ─────────────────────────────────────────────────
    legend_handles = {}

    for spot in candidates:
        phase = spot.get("phase_label", "unknown")
        color = phase_color[phase]
        E_eV  = float(spot["E"])
        kf_hat = _kf_hat_from_spot(spot)
        cos_out = max(abs(float(np.dot(stack.n_hat, kf_hat))), 1e-3)

        # Build depth samples: (z_from_surface_mm, absorption_weight)
        # Only sample depths within the layer that produced this spot —
        # T_above is still accumulated through ALL overlying layers so that
        # absorption by the film is correctly applied to substrate spots.
        depth_samples = []   # (z_mm, weight)
        T_above = 1.0        # cumulative transmission from surface to current layer top
        spot_phase = spot.get("phase_label")

        for z_start_ang, z_end_ang, layer in segments:
            thick_mm = (z_end_ang - z_start_ang) * 1e-7  # Å → mm
            mu = layer._linear_mu(E_eV) * 1e7             # Å⁻¹ → mm⁻¹

            # Only collect depth samples from the layer that produced this spot
            if spot_phase is None or layer.label == spot_phase:
                n_samp = max(1, n_steps_per_layer)
                zs_rel = np.linspace(0.0, thick_mm, n_samp + 1)[:-1]
                for dz in zs_rel:
                    z_mm = z_start_ang * 1e-7 + dz
                    T_partial = (np.exp(-mu * dz * (1.0 / cos_in + 1.0 / cos_out))
                                 if mu > 0 else 1.0)
                    depth_samples.append((z_mm, T_above * T_partial))

            # Always advance T_above through every layer (correct absorption)
            if mu > 0:
                T_above *= np.exp(-mu * thick_mm * (1.0 / cos_in + 1.0 / cos_out))

            # Stop early once signal is negligible
            if T_above < 1e-4:
                break

        if not depth_samples:
            continue

        # Project each depth sample onto the detector
        pix_trail = []
        weights   = []
        for z_mm, w in depth_samples:
            # source_depth_mm: displacement along beam = depth × cos_in
            src_mm = z_mm / cos_in
            if space == "detector":
                pix = camera.project(kf_hat, source_depth_mm=src_mm)
                if pix is not None:
                    pix_trail.append(pix)
                    weights.append(w)
            else:
                # angles space: depth doesn't shift 2θ/χ — only pixel position
                # We still show the surface point only for clarity
                pix_trail.append((float(spot["tth"]), float(spot["chi"])))
                weights.append(w)
                break  # single point in angle space

        if not pix_trail:
            continue

        pix_trail = np.array(pix_trail)
        weights   = np.array(weights)
        weights  /= weights.max()

        xs = pix_trail[:, 0]
        ys = pix_trail[:, 1]

        # Draw the trail as a sequence of line segments with fading alpha
        if len(xs) > 1:
            points  = np.column_stack([xs, ys]).reshape(-1, 1, 2)
            segs    = np.concatenate([points[:-1], points[1:]], axis=1)
            alphas  = 0.1 + 0.85 * weights[:-1]   # surface → opaque, deep → faint
            rgba    = np.array([(*mcolors.to_rgb(color), a) for a in alphas])
            lc = LineCollection(segs, colors=rgba, linewidths=1.5, zorder=2)
            ax.add_collection(lc)

        # Trail endpoints
        ax.scatter(xs[0], ys[0], s=10, color=color, alpha=0.4,
                   zorder=3, marker="|")
        ax.scatter(xs[-1], ys[-1], s=10, color=color, alpha=0.4,
                   zorder=3, marker="|")

        # Simulation spot position — Circle in data coords so it grows on zoom
        if space == "angles":
            sim_x, sim_y = float(spot["tth"]), float(spot["chi"])
        else:
            sim_x, sim_y = float(spot["pix"][0]), float(spot["pix"][1])
        ax.add_patch(Circle((sim_x, sim_y), radius=1.5, facecolor=color,
                            edgecolor="white", linewidth=0.5, alpha=0.5, zorder=5))

        # Combined ellipse: depth-parallax covariance + beam-divergence covariance.
        # Centred at the absorption-weighted mean of the trail so it spans
        # the same region as the trail itself.
        if show_divergence and space == "detector" and len(xs) > 1:
            from matplotlib.patches import Ellipse as _Ellipse
            w_norm = weights / weights.sum()
            mx = float(np.dot(w_norm, xs))
            my = float(np.dot(w_norm, ys))
            dx = xs - mx
            dy = ys - my
            cov_depth = np.array([
                [float(np.dot(w_norm, dx * dx)), float(np.dot(w_norm, dx * dy))],
                [float(np.dot(w_norm, dx * dy)), float(np.dot(w_norm, dy * dy))],
            ])
            cov_div = np.asarray(spot.get("cov_px", np.zeros((2, 2))), dtype=float)
            cov_total = cov_depth + cov_div
            try:
                eigvals, eigvecs = np.linalg.eigh(cov_total)
                eigvals = np.maximum(eigvals, 0.0)
                idx = int(np.argmax(eigvals))
                sig_maj = float(np.sqrt(eigvals[idx]))
                sig_min = float(np.sqrt(eigvals[1 - idx]))
                if sig_maj > 0.0:
                    v = eigvecs[:, idx]
                    ang = float(np.degrees(np.arctan2(v[1], v[0])))
                    ax.add_patch(_Ellipse(
                        xy=(mx, my),
                        width=2.0 * divergence_nsigma * sig_maj,
                        height=2.0 * divergence_nsigma * sig_min,
                        angle=ang,
                        linewidth=0.8, edgecolor=color, facecolor="none",
                        alpha=0.6, zorder=3,
                    ))
            except Exception:
                pass

        # hkl annotation at the simulation spot position
        h, k, l = spot["hkl"]
        ax.annotate(
            f"({h}{k}{l})",
            (sim_x, sim_y),
            xytext=(4, 4), textcoords="offset points",
            fontsize=5, color=color, alpha=0.85, zorder=6,
        )

        if phase not in legend_handles:
            legend_handles[phase] = plt.Line2D(
                [], [], color=color, linewidth=2,
                label=phase, marker="o", markersize=4,
            )

    # ── Axes decoration ───────────────────────────────────────────────────────
    if space == "detector":
        ax.set_xlabel("x  (px)", color=FG, fontsize=9)
        ax.set_ylabel("y  (px)", color=FG, fontsize=9)
        ax.set_aspect("equal")
    else:
        ax.set_xlabel("2θ  (°)", color=FG, fontsize=9)
        ax.set_ylabel("χ  (°)",  color=FG, fontsize=9)

    depth_lim_um = min(total_thickness_mm * 1e3,
                       max(seg[1] - seg[0] for seg in segments) * 1e-4)
    ax.set_title(
        f"Depth-parallax elongation  "
        f"(total stack {total_thickness_mm*1e3:.0f} µm, "
        f"surface → deep,  cos_in = {cos_in:.3f})",
        color=FG, fontsize=8,
    )

    if legend_handles:
        leg = ax.legend(
            handles=list(legend_handles.values()),
            fontsize=7, labelcolor=FG,
            facecolor="#1a1f2e", edgecolor="#333355",
            loc="upper right",
        )

    if standalone:
        fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {out_path}")

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 3D UNIT CELL + HKL PLANE
# ─────────────────────────────────────────────────────────────────────────────

_CELL_CORNERS_FRAC = np.array([
    [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1],
], dtype=float)


def _clip_polygon_axis(poly, axis, bound, keep_min):
    """Sutherland-Hodgman clip of a planar polygon against one axis-aligned
    half-space (`coord >= bound` if `keep_min` else `coord <= bound`)."""
    if len(poly) == 0:
        return poly
    out = []
    n = len(poly)
    for i in range(n):
        curr, prev = poly[i], poly[i - 1]
        curr_in = (curr[axis] >= bound) if keep_min else (curr[axis] <= bound)
        prev_in = (prev[axis] >= bound) if keep_min else (prev[axis] <= bound)
        if curr_in != prev_in:
            t = (bound - prev[axis]) / (curr[axis] - prev[axis])
            out.append(prev + t * (curr - prev))
        if curr_in:
            out.append(curr)
    return out


def _hkl_plane_fractional(h, k, l, m):
    """Intersection polygon (fractional coords) of the plane h·u+k·v+l·w=m
    with the unit cell cube [0,1]³, as a list of ordered vertices."""
    n = np.array([h, k, l], dtype=float)
    if h != 0:
        p0 = np.array([m / h, 0.0, 0.0])
    elif k != 0:
        p0 = np.array([0.0, m / k, 0.0])
    else:
        p0 = np.array([0.0, 0.0, m / l])

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(n, tmp)) > 0.9 * np.linalg.norm(n):
        tmp = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, tmp)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    e2 /= np.linalg.norm(e2)

    big = 5.0  # spans well beyond the unit cube before clipping
    poly = [p0 + big * e1 + big * e2, p0 - big * e1 + big * e2,
            p0 - big * e1 - big * e2, p0 + big * e1 - big * e2]
    for axis in range(3):
        poly = _clip_polygon_axis(poly, axis, 0.0, keep_min=True)
        poly = _clip_polygon_axis(poly, axis, 1.0, keep_min=False)
    return poly


def _normalize_hkl_list(hkl):
    """Accept a single `(h, k, l)` or a list of them; always return a list."""
    if len(hkl) == 3 and all(np.isscalar(v) for v in hkl):
        return [tuple(int(v) for v in hkl)]
    return [tuple(int(v) for v in item) for item in hkl]


def plot_unit_cell(
    crystal,
    hkl: "tuple[int, int, int] | list[tuple[int, int, int]]" = (1, 1, 1),
    *,
    plane_offset: "int | list[int | None] | None" = None,
    show_normal: bool = True,
    show_axes_arrows: bool = True,
    cell_color: str = FG,
    plane_colors: "str | list[str] | None" = None,
    normal_color: str = COL_DB,
    elev: float = 20.0,
    azim: float = -50.0,
    figsize: "tuple[float, float]" = (7, 7),
    ax: "plt.Axes | None" = None,
    out_path: "str | None" = None,
):
    """
    Draw the full (conventional, non-reduced) unit cell of *crystal* in 3-D
    and overlay one or more representative planes from the `hkl` families
    for visualization.

    The cell is the parallelepiped spanned by `crystal.a1`, `crystal.a2`,
    `crystal.a3` (the real-space lattice vectors exactly as built by
    `xrayutilities.materials.SGLattice`/`Crystal` — no Niggli/Delaunay
    reduction is applied, matching what CIF or `build_bcc`/`build_b2` etc.
    hand back). Each `(hkl)` plane is found as the intersection of
    `h·u + k·v + l·w = m` (fractional coordinates `u,v,w`, the textbook
    Miller-index construction) with the unit cell cube, then mapped back to
    Cartesian space through `crystal.a1/a2/a3`. `m` (the plane's distance
    from the origin, in lattice-plane units) defaults to whichever of ±1
    actually intersects the cell interior, so both all-positive and
    mixed-sign index families render a sensible cross-section.

    Args:
        crystal: An *xrayutilities*-compatible crystal object exposing
            `.a1 .a2 .a3` (Cartesian real-space cell vectors, Å),
            `.Q(h, k, l)` (reciprocal-lattice vector) and
            `.planeDistance(h, k, l)` (d-spacing).
        hkl: Miller indices of the plane to draw, e.g. `(1, 1, 1)`, or a
            list of several, e.g. `[(1, 1, 1), (1, 0, 0)]`.
        plane_offset: Override the automatic choice of `m` above (e.g. `2`
            to draw the second plane of the family instead of the first).
            Either a single value applied to every plane in `hkl`, or a
            list matching `hkl` one-to-one.
        show_normal: Draw an arrow along `crystal.Q(hkl)` (the plane
            normal), anchored at that plane's own centroid.
        show_axes_arrows: Draw labelled `a`, `b`, `c` arrows from the
            origin.
        cell_color, normal_color: Line colours.
        plane_colors: Fill/edge colour(s) for the plane(s). A single colour
            applied to all planes, a list matching `hkl` one-to-one, or
            `None` to cycle through the project's standard grain palette.
        elev, azim: 3-D view angle (degrees), forwarded to `Axes3D.view_init`.
        figsize: Figure size in inches (only used when `ax is None`).
        ax: Draw into an existing 3-D :class:`~matplotlib.axes.Axes`
            (created with `projection='3d'`); `None` creates a new figure.
        out_path: Save path; `None` → do not save.

    Returns:
        `(fig, ax)`

    Example:
    >>> crystal = crystal_from_cif("Al2O3.cif")
    >>> plot_unit_cell(crystal, hkl=[(1, 0, 4), (0, 0, 1)], out_path="al2o3.png")
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    hkls = _normalize_hkl_list(hkl)
    if any(h == 0 and k == 0 and l == 0 for h, k, l in hkls):
        raise ValueError("hkl=(0, 0, 0) does not define a plane")

    n_planes = len(hkls)
    offsets = plane_offset if isinstance(plane_offset, (list, tuple)) else [plane_offset] * n_planes
    if len(offsets) != n_planes:
        raise ValueError("plane_offset list must match the number of hkl planes")
    if plane_colors is None:
        colors = [_GRAIN_COLORS[i % len(_GRAIN_COLORS)] for i in range(n_planes)]
    elif isinstance(plane_colors, str):
        colors = [plane_colors] * n_planes
    else:
        colors = list(plane_colors)
        if len(colors) != n_planes:
            raise ValueError("plane_colors list must match the number of hkl planes")

    ai = np.array([crystal.a1, crystal.a2, crystal.a3], dtype=float)
    cart_corners = _CELL_CORNERS_FRAC @ ai
    edges = [
        (i, j)
        for i in range(8)
        for j in range(i + 1, 8)
        if np.sum(_CELL_CORNERS_FRAC[i] != _CELL_CORNERS_FRAC[j]) == 1
    ]

    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.03, 0.05, 0.08, 1.0))
        pane._axinfo["grid"]["color"] = (0.1, 0.12, 0.18, 0.6)
    ax.tick_params(colors="#7788aa", labelsize=7)

    for i, j in edges:
        p, q = cart_corners[i], cart_corners[j]
        ax.plot(*zip(p, q), color=cell_color, lw=1.2, alpha=0.6)
    ax.scatter(*cart_corners.T, color=cell_color, s=12, alpha=0.8)

    legend_handles = []
    drew_any_normal = False
    normal_arrow_len = 0.6 * min(np.linalg.norm(v) for v in ai)
    for (h, k, l), user_offset, plane_color in zip(hkls, offsets, colors):
        if user_offset is not None:
            offsets_to_try = [user_offset]
        else:
            f_max = max(h, 0) + max(k, 0) + max(l, 0)
            offsets_to_try = [1, -1] if f_max >= 1 else [-1, 1]

        frac_poly = []
        for m in offsets_to_try:
            frac_poly = _hkl_plane_fractional(h, k, l, m)
            if len(frac_poly) >= 3:
                break

        d_hkl = crystal.planeDistance(h, k, l)
        if len(frac_poly) < 3:
            print(f"  Warning: ({h}{k}{l}) plane does not intersect the unit "
                  f"cell for the offsets tried ({offsets_to_try}); pass "
                  f"plane_offset explicitly to pick a different member of "
                  f"the family.")
            continue

        cart_poly = np.array(frac_poly) @ ai
        ax.add_collection3d(Poly3DCollection(
            [cart_poly], facecolor=plane_color, edgecolor=plane_color,
            linewidths=1.2, alpha=0.35,
        ))
        legend_handles.append(
            Line2D([0], [0], color=plane_color, lw=2,
                   label=f"({h}{k}{l})  d = {d_hkl:.4f} Å")
        )

        if show_normal:
            # Anchored at this plane's own centroid (not the cell centroid)
            # so the arrow sits flush against the plane it belongs to.
            plane_centroid = cart_poly.mean(axis=0)
            n_hat = crystal.Q(h, k, l)
            n_hat = n_hat / np.linalg.norm(n_hat)
            ax.quiver(*plane_centroid, *(n_hat * normal_arrow_len),
                       color=normal_color, linewidth=1.8, arrow_length_ratio=0.15)
            drew_any_normal = True

    if drew_any_normal:
        legend_handles.append(
            Line2D([0], [0], color=normal_color, lw=2, label="plane normal (Q_hkl)")
        )

    if show_axes_arrows:
        for vec, label in zip(ai, ("a", "b", "c")):
            ax.quiver(0, 0, 0, *vec, color=cell_color, linewidth=1.0,
                      arrow_length_ratio=0.08, alpha=0.9)
            ax.text(*(vec * 1.08), label, color=cell_color, fontsize=10)

    ax.set_xlabel("x  (Å)", color=FG, fontsize=8, labelpad=8)
    ax.set_ylabel("y  (Å)", color=FG, fontsize=8, labelpad=8)
    ax.set_zlabel("z  (Å)", color=FG, fontsize=8, labelpad=8)
    title = ", ".join(f"({h}{k}{l})" for h, k, l in hkls) if n_planes <= 3 else f"{n_planes} planes"
    ax.set_title(f"Unit cell — {title}", color=FG, fontsize=10, pad=10)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect(np.ptp(cart_corners, axis=0))
    except AttributeError:
        pass

    if legend_handles:
        ax.legend(
            handles=legend_handles, fontsize=7, labelcolor=FG,
            facecolor="#1a1f2e", edgecolor="#333355", loc="upper left",
        )

    if standalone:
        fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {out_path}")

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 3D UNIT CELL IN THE LAB FRAME
# ─────────────────────────────────────────────────────────────────────────────

_KI_HAT = np.array([1.0, 0.0, 0.0])   # incident-beam direction, LT lab frame


def _bragg_kf_hat(G_lab, ki_hat=_KI_HAT):
    """
    Direction of the elastically diffracted beam for reciprocal-lattice
    vector `G_lab` (lab frame) given an incident beam along `ki_hat`, or
    `None` if no elastic solution exists (`G_lab · ki_hat >= 0` -- this
    reflection is never excited for a beam along `ki_hat`, at any energy).

    Solves `|k0·ki_hat + G_lab| = k0` for `k0`, the same convention used
    throughout `nrxrdct.laue.simulation` (e.g. the harmonic search in
    `beam_divergence_ellipses`).
    """
    Gm2 = float(np.dot(G_lab, G_lab))
    kdG = float(np.dot(ki_hat, G_lab))
    if kdG >= 0:
        return None
    lam = -4.0 * np.pi * kdG / Gm2
    k0 = 2.0 * np.pi / lam
    kf = k0 * ki_hat + G_lab
    return kf / np.linalg.norm(kf)


def plot_unit_cell_in_lab(
    crystal,
    U: "np.ndarray",
    hkl: "tuple[int, int, int] | list[tuple[int, int, int]]" = (1, 1, 1),
    *,
    camera: "Camera | None" = None,
    plane_offset: "int | list[int | None] | None" = None,
    show_normal: bool = True,
    show_crystal_axes: bool = True,
    show_lab_axes: bool = True,
    show_beam: bool = True,
    show_scattered_rays: bool = True,
    cell_color: str = FG,
    plane_colors: "str | list[str] | None" = None,
    normal_color: str = COL_DB,
    beam_color: str = "red",
    elev: float = 0.0,
    azim: float = -90.0,
    figsize: "tuple[float, float]" = (8, 8),
    ax: "plt.Axes | None" = None,
    out_path: "str | None" = None,
):
    """
    Like :func:`plot_unit_cell`, but rotates the crystal's unit cell (and
    `hkl` plane(s)) into the lab frame via the orientation matrix `U`, and
    adds the lab-frame furniture from :func:`plot_layer_scheme` -- incident
    beam, x/y/z lab axes and each hkl's scattered-ray direction -- rendered
    in 3-D instead of a 2-D XZ cross-section.

    Lab frame convention (LaueTools / this project, matching
    `plot_layer_scheme`): x = incident beam direction, z = vertical up,
    y completes the right-handed frame. `v_lab = U @ v_crystal` for any
    crystal-frame vector (cell edges, `crystal.Q(hkl)`, etc.) -- the same
    convention used throughout `nrxrdct.laue.simulation`
    (e.g. `layer.U @ layer.crystal.Q(h, k, l)`).

    The incident beam and every scattered ray are drawn touching the same
    point: the centroid of the first `hkl` plane that was actually drawn
    (the same point its normal arrow is anchored to), falling back to the
    cell centroid if no plane intersects the cell.

    Each `hkl`'s elastically diffracted direction is found by solving the
    Laue condition `|k0·x̂ + G_lab| = k0` (see `_bragg_kf_hat`) and drawn as
    a ray of schematic length from the scattering point -- reflections with
    no elastic solution for a beam along +x are skipped with a printed
    note. Passing a `camera` adds the detail of *where it actually lands*:
    each ray is projected with `camera.project` and drawn solid when it
    hits the active detector area, dashed when it doesn't (annotated with
    the pixel coordinate). No detector geometry is drawn -- the cell is
    Å-sized and `camera.dd` is tens of millimetres, so a to-scale detector
    would either vanish or swallow the cell.

    Args:
        crystal: xrayutilities-compatible crystal object (see `plot_unit_cell`).
        U: `(3, 3)` orientation matrix rotating crystal-frame vectors into
            the lab frame.
        hkl: Miller indices of the plane(s) to draw (single tuple or list).
        camera: Optional :class:`~nrxrdct.laue.camera.Camera`. When given,
            scattered rays are drawn solid/dashed by whether they land on
            the detector, and labelled with the projected pixel coordinate.
        plane_offset, show_normal, plane_colors, normal_color: see
            `plot_unit_cell`.
        show_crystal_axes: Draw labelled `a`, `b`, `c` arrows (rotated into
            the lab frame) from the cell origin.
        show_lab_axes: Draw the fixed lab-frame x/y/z axes near the cell.
        show_beam: Draw the incident-beam arrow travelling along +x, tip
            touching the scattering point.
        show_scattered_rays: Draw each hkl's diffracted-ray direction (works
            with or without `camera`).
        cell_color, beam_color: Line colours.
        elev, azim: 3-D view angle (degrees), forwarded to `Axes3D.view_init`.
            Defaults to a view looking down +y (elev=0, azim=-90), so the
            beam (x) / up (z) plane is seen face-on.
        figsize: Figure size in inches (only used when `ax is None`).
        ax: Draw into an existing 3-D :class:`~matplotlib.axes.Axes`
            (created with `projection='3d'`); `None` creates a new figure.
        out_path: Save path; `None` → do not save.

    Returns:
        `(fig, ax)`

    Example:
    >>> U = euler_to_U(10, 20, 30)
    >>> plot_unit_cell_in_lab(crystal, U, hkl=[(1, 1, 1), (1, 0, 0)], camera=cam)
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    U = np.asarray(U, dtype=float)
    hkls = _normalize_hkl_list(hkl)
    if any(h == 0 and k == 0 and l == 0 for h, k, l in hkls):
        raise ValueError("hkl=(0, 0, 0) does not define a plane")

    n_planes = len(hkls)
    offsets = plane_offset if isinstance(plane_offset, (list, tuple)) else [plane_offset] * n_planes
    if len(offsets) != n_planes:
        raise ValueError("plane_offset list must match the number of hkl planes")
    if plane_colors is None:
        colors = [_GRAIN_COLORS[i % len(_GRAIN_COLORS)] for i in range(n_planes)]
    elif isinstance(plane_colors, str):
        colors = [plane_colors] * n_planes
    else:
        colors = list(plane_colors)
        if len(colors) != n_planes:
            raise ValueError("plane_colors list must match the number of hkl planes")

    ai_cry = np.array([crystal.a1, crystal.a2, crystal.a3], dtype=float)
    ai = ai_cry @ U.T   # rows a1,a2,a3 rotated into the lab frame
    cart_corners = _CELL_CORNERS_FRAC @ ai
    edges = [
        (i, j)
        for i in range(8)
        for j in range(i + 1, 8)
        if np.sum(_CELL_CORNERS_FRAC[i] != _CELL_CORNERS_FRAC[j]) == 1
    ]

    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.03, 0.05, 0.08, 1.0))
        pane._axinfo["grid"]["color"] = (0.1, 0.12, 0.18, 0.6)
    ax.tick_params(colors="#7788aa", labelsize=7)

    for i, j in edges:
        p, q = cart_corners[i], cart_corners[j]
        ax.plot(*zip(p, q), color=cell_color, lw=1.2, alpha=0.6)
    ax.scatter(*cart_corners.T, color=cell_color, s=12, alpha=0.8)

    legend_handles = []
    drew_any_normal = False
    G_labs = []                 # one entry per hkl, aligned with hkls/colors
    scatter_pt = None           # centroid of the first successfully-drawn plane
    normal_arrow_len = 0.6 * min(np.linalg.norm(v) for v in ai)
    for (h, k, l), user_offset, plane_color in zip(hkls, offsets, colors):
        G_lab = U @ crystal.Q(h, k, l)
        G_labs.append(G_lab)

        if user_offset is not None:
            offsets_to_try = [user_offset]
        else:
            f_max = max(h, 0) + max(k, 0) + max(l, 0)
            offsets_to_try = [1, -1] if f_max >= 1 else [-1, 1]

        frac_poly = []
        for m in offsets_to_try:
            frac_poly = _hkl_plane_fractional(h, k, l, m)
            if len(frac_poly) >= 3:
                break

        d_hkl = crystal.planeDistance(h, k, l)
        if len(frac_poly) < 3:
            print(f"  Warning: ({h}{k}{l}) plane does not intersect the unit "
                  f"cell for the offsets tried ({offsets_to_try}); pass "
                  f"plane_offset explicitly to pick a different member of "
                  f"the family.")
            continue

        cart_poly = np.array(frac_poly) @ ai
        ax.add_collection3d(Poly3DCollection(
            [cart_poly], facecolor=plane_color, edgecolor=plane_color,
            linewidths=1.2, alpha=0.35,
        ))
        legend_handles.append(
            Line2D([0], [0], color=plane_color, lw=2,
                   label=f"({h}{k}{l})  d = {d_hkl:.4f} Å")
        )

        # Anchored at this plane's own centroid, and rotated by U along with
        # everything else -- see plot_unit_cell for why the centroid (not
        # the cell centroid) is the correct anchor. The first plane drawn
        # also becomes the common scattering point for the beam/rays below.
        plane_centroid = cart_poly.mean(axis=0)
        if scatter_pt is None:
            scatter_pt = plane_centroid
        if show_normal:
            n_hat = G_lab / np.linalg.norm(G_lab)
            ax.quiver(*plane_centroid, *(n_hat * normal_arrow_len),
                       color=normal_color, linewidth=1.8, arrow_length_ratio=0.15)
            drew_any_normal = True

    if drew_any_normal:
        legend_handles.append(
            Line2D([0], [0], color=normal_color, lw=2, label="plane normal (Q_hkl)")
        )

    if show_crystal_axes:
        for vec, label in zip(ai, ("a", "b", "c")):
            ax.quiver(0, 0, 0, *vec, color=cell_color, linewidth=1.0,
                      arrow_length_ratio=0.08, alpha=0.9)
            ax.text(*(vec * 1.08), label, color=cell_color, fontsize=10)

    if scatter_pt is None:
        scatter_pt = cart_corners.mean(axis=0)

    # ── Lab furniture: incident beam (+x), x/y/z axes, scattered rays ─────────
    # Mirrors the beam arrow / axis triad drawn by plot_layer_scheme, just in
    # 3-D and anchored relative to the (possibly tilted) cell's own extent.
    cell_span = np.ptp(cart_corners, axis=0)
    scale = float(cell_span.max()) if cell_span.max() > 0 else 1.0

    if show_beam:
        # Tip touches the scattering point -- the same anchor used for the
        # plane normal(s) above -- rather than merely approaching the cell.
        beam_tip = scatter_pt
        beam_tail = beam_tip - np.array([0.8 * scale, 0.0, 0.0])
        ax.quiver(*beam_tail, *(beam_tip - beam_tail), color=beam_color,
                  linewidth=2.2, arrow_length_ratio=0.1)
        ax.text(*beam_tail, "incident beam", color=beam_color, fontsize=8,
                ha="left", va="bottom")
        legend_handles.append(
            Line2D([0], [0], color=beam_color, lw=2, label="incident beam (+x)")
        )

    if show_scattered_rays:
        ray_length = 1.2 * scale
        drew_any_ray = False
        for (h, k, l), G_lab, plane_color in zip(hkls, G_labs, colors):
            kf_hat = _bragg_kf_hat(G_lab)
            if kf_hat is None:
                print(f"  Note: ({h}{k}{l}) is not excited for a beam along "
                      f"+x (G·x̂ ≥ 0) -- no elastic scattered ray to draw.")
                continue
            ray_end = scatter_pt + kf_hat * ray_length
            if camera is not None:
                pix = camera.project(kf_hat, source_depth_mm=0.0)
                linestyle = "-" if pix is not None else "--"
                label = (f"({h}{k}{l}) → px({pix[0]:.0f}, {pix[1]:.0f})"
                          if pix is not None else f"({h}{k}{l}) → off detector")
            else:
                linestyle = "-"
                label = f"({h}{k}{l})"
            ax.plot(*zip(scatter_pt, ray_end), color=plane_color, lw=1.4,
                    linestyle=linestyle, alpha=0.85)
            ax.text(*ray_end, label, color=plane_color, fontsize=7)
            drew_any_ray = True
        if drew_any_ray:
            ray_legend_label = (
                "scattered ray  (— on det.,  -- off det.)"
                if camera is not None else "scattered ray"
            )
            legend_handles.append(
                Line2D([0], [0], color=FG, lw=1.4, linestyle="-", label=ray_legend_label)
            )

    if show_lab_axes:
        origin = cart_corners.min(axis=0) - 0.6 * scale
        ax_len = 0.5 * scale
        for direction, color, label in (
            (np.array([1.0, 0.0, 0.0]), "#4fc3f7", "x (beam)"),
            (np.array([0.0, 1.0, 0.0]), "#88cc88", "y"),
            (np.array([0.0, 0.0, 1.0]), "#ff9f43", "z (up)"),
        ):
            tip = origin + ax_len * direction
            ax.quiver(*origin, *(tip - origin), color=color, linewidth=2.0,
                      arrow_length_ratio=0.2)
            ax.text(*(tip + 0.1 * ax_len * direction), label, color=color,
                    fontsize=9, fontweight="bold")

    ax.set_xlabel("x  (Å)", color=FG, fontsize=8, labelpad=8)
    ax.set_ylabel("y  (Å)", color=FG, fontsize=8, labelpad=8)
    ax.set_zlabel("z  (Å)", color=FG, fontsize=8, labelpad=8)
    title = ", ".join(f"({h}{k}{l})" for h, k, l in hkls) if n_planes <= 3 else f"{n_planes} planes"
    ax.set_title(f"Unit cell in lab frame — {title}", color=FG, fontsize=10, pad=10)
    ax.view_init(elev=elev, azim=azim)
    try:
        ax.set_box_aspect(cell_span)
    except AttributeError:
        pass

    if legend_handles:
        ax.legend(
            handles=legend_handles, fontsize=7, labelcolor=FG,
            facecolor="#1a1f2e", edgecolor="#333355", loc="upper left",
        )

    if standalone:
        fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {out_path}")

    return fig, ax


# ─────────────────────────────────────────────────────────────────────────────
# 3D Q-SPACE RECONSTRUCTION AROUND A SPOT
# ─────────────────────────────────────────────────────────────────────────────

_QVOL_LOW = "#140a04"    # near-black warm — blends into BG, marks ~zero intensity
_QVOL_MID = COL_SUP      # orange — matches the existing satellite/superstructure accent
_QVOL_HIGH = COL_DB      # pale yellow — matches the existing direct-beam accent
_QVOL_WINDOW = COL_BCC   # translucent surface marking the energy-window boundary


def plot_qspace_around_spot(
    vol: dict,
    *,
    log_intensity: bool = False,
    intensity_floor: float = 1e-6,
    show_window_surface: bool = True,
    show_off_detector: bool = True,
    elev: float = 22.0,
    azim: float = -60.0,
    figsize: tuple[float, float] = (9, 8),
    ax: "plt.Axes | None" = None,
    out_path: "str | None" = None,
):
    """
    Render the 3-D reciprocal-space volume returned by
    :func:`~nrxrdct.laue.simulation.qspace_around_spot` and overlay the
    boundary of the accessible white-beam energy window, to show why some
    fringe/satellite points along the rod turn into detector spots and
    others don't.

    Three things have to go right before a voxel of the reconstructed
    volume can appear as a spot on the detector image, and this plot makes
    all three visible at once:

    1. **Structure-factor intensity** — `vol['I']`.  Destructive
       interference along the rod (finite-thickness fringes, superlattice
       satellites) drives some voxels to ~0 even though they are otherwise
       fully reachable.  Encoded as marker colour + size (sequential warm
       ramp: near-black → orange → pale yellow, matching this project's
       existing satellite / direct-beam accent colours).
    2. **Energy window** — a voxel needs a photon energy inside
       `[E_min_eV, E_max_eV]` to be excited at all.  Drawn as a translucent
       blue surface: the boundary where `vol['reachable']` flips off.  Rod
       points beyond this surface cannot appear on any detector, at any
       intensity, in this beam.
    3. **Detector footprint** — even within the energy window, the
       diffracted ray must land on the physical detector.  Voxels that are
       reachable but off-detector (`vol['on_detector'] == False`) are drawn
       as small hollow markers instead of filled ones.

    Args:
        vol: Return value of
            :func:`~nrxrdct.laue.simulation.qspace_around_spot`.
        log_intensity: Colour/size by `log10(I / I_max + floor)` instead of
            the raw ratio.  Structure-factor intensities span many orders
            of magnitude near a fringe minimum, so this is `True` by default.
        intensity_floor: Added (as a fraction of the volume's max
            intensity) before taking the log, so exact zeros stay finite.
        show_window_surface: Draw the translucent energy-window boundary
            surface.  It is only drawn where the boundary actually falls
            inside the grid — a box entirely inside or entirely outside the
            window has no boundary to show (widen `extent_along` /
            `max_satellites` in `qspace_around_spot` to bring it into view).
        show_off_detector: Draw reachable-but-off-detector voxels as hollow
            markers.  Has no effect if `vol` was built without a `camera`
            (no on/off-detector information exists in that case).
        elev, azim: 3-D view angle (degrees), forwarded to `Axes3D.view_init`.
        figsize: Figure size in inches (only used when `ax is None`).
        ax: Draw into an existing 3-D :class:`~matplotlib.axes.Axes`
            (created with `projection='3d'`); `None` creates a new figure.
        out_path: Save path; `None` → do not save.

    Returns:
        `(fig, ax)`

    Example:
    >>> vol = qspace_around_spot(stack, (0, 0, 2), camera=cam)
    >>> plot_qspace_around_spot(vol, out_path="qspace_002.png")
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers '3d' projection

    along = np.asarray(vol["along"])
    lat1 = np.asarray(vol["lateral1"])
    lat2 = np.asarray(vol["lateral2"])
    I = vol["I"]
    reachable = vol["reachable"]
    on_det = vol["on_detector"]
    shape = I.shape

    AA, T1, T2 = np.meshgrid(along, lat1, lat2, indexing="ij")

    # ── colour / size by intensity ────────────────────────────────────────────
    Imax = float(I.max()) if I.size and I.max() > 0 else 1.0
    if log_intensity:
        val = np.log10(I / Imax + intensity_floor)
        vmin, vmax = np.log10(intensity_floor), 0.0
    else:
        val = I / Imax
        vmin, vmax = 0.0, 1.0

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "qvol_warm", [_QVOL_LOW, _QVOL_MID, _QVOL_HIGH]
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    frac = np.clip((val - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    sizes = 3.0 + 40.0 * frac ** 1.5

    # ── which voxels get a marker at all ──────────────────────────────────────
    has_camera = on_det is not None
    if has_camera:
        filled = on_det                       # on_detector already implies reachable
        hollow = reachable & ~on_det
    else:
        filled = reachable
        hollow = np.zeros(shape, dtype=bool)

    # ── set up axes ────────────────────────────────────────────────────────────
    standalone = ax is None
    if standalone:
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor(BG)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((0.03, 0.05, 0.08, 1.0))
        pane._axinfo["grid"]["color"] = (0.1, 0.12, 0.18, 0.6)
    ax.tick_params(colors="#7788aa", labelsize=7)

    sc = ax.scatter(
        AA[filled], T1[filled], T2[filled],
        c=val[filled], cmap=cmap, norm=norm,
        s=sizes[filled], marker="o", linewidths=0, alpha=0.9, depthshade=True,
    )

    legend_handles = []
    if show_off_detector and has_camera and hollow.any():
        edge_colors = cmap(norm(val[hollow]))
        ax.scatter(
            AA[hollow], T1[hollow], T2[hollow],
            s=np.clip(sizes[hollow], 6, None), marker="o",
            facecolors="none", edgecolors=edge_colors,
            linewidths=0.6, alpha=0.85,
        )
        legend_handles.append(
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="none",
                   markeredgecolor=FG, markersize=7,
                   label="reachable, off detector")
        )

    # ── energy-window boundary surface ────────────────────────────────────────
    if show_window_surface:
        # For each (lateral1, lateral2) column, find where `reachable` flips
        # off along the rod axis. A column that is entirely True or entirely
        # False has no boundary within this grid and is left as NaN.
        n_lat1, n_lat2 = shape[1], shape[2]
        drew_surface = False
        for edge in ("lo", "hi"):
            boundary = np.full((n_lat1, n_lat2), np.nan)
            for j in range(n_lat1):
                for k in range(n_lat2):
                    col = reachable[:, j, k]
                    if not col.any() or col.all():
                        continue
                    idxs = np.where(col)[0]
                    boundary[j, k] = along[idxs.min() if edge == "lo" else idxs.max()]
            if np.isnan(boundary).all():
                continue
            T1g, T2g = np.meshgrid(lat1, lat2, indexing="ij")
            ax.plot_surface(
                np.ma.masked_invalid(boundary), T1g, T2g,
                color=_QVOL_WINDOW, alpha=0.18, linewidth=0, shade=False,
            )
            drew_surface = True
        if drew_surface:
            legend_handles.append(
                Line2D([0], [0], marker="s", linestyle="", color=_QVOL_WINDOW,
                       alpha=0.5, markersize=9,
                       label="energy-window boundary (E_min/E_max)")
            )

    # ── labels / colorbar / legend ────────────────────────────────────────────
    hkl = vol.get("hkl")
    layer_label = vol.get("layer")
    ax.set_xlabel("ΔQ ∥ n̂  (Å⁻¹)", color=FG, fontsize=8, labelpad=8)
    ax.set_ylabel("ΔQ transverse 1  (Å⁻¹)", color=FG, fontsize=8, labelpad=8)
    ax.set_zlabel("ΔQ transverse 2  (Å⁻¹)", color=FG, fontsize=8, labelpad=8)
    ax.set_title(
        f"Q-space reconstruction around hkl={hkl}  (layer '{layer_label}')",
        color=FG, fontsize=10, pad=10,
    )
    ax.view_init(elev=elev, azim=azim)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("log₁₀ I / I_max" if log_intensity else "I / I_max", color=FG, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#7788aa", labelsize=7)
    plt.setp(cbar.ax.get_yticklabels(), color=FG)
    cbar.outline.set_edgecolor("#333355")

    if legend_handles:
        ax.legend(
            handles=legend_handles, fontsize=7, labelcolor=FG,
            facecolor="#1a1f2e", edgecolor="#333355", loc="upper left",
        )

    if standalone:
        fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {out_path}")

    return fig, ax


def plot_qspace_on_detector(
    vol: dict,
    *,
    camera: "Camera | None" = None,
    image: "np.ndarray | None" = None,
    log_intensity: bool = False,
    intensity_floor: float = 1e-6,
    figsize: tuple[float, float] = (8, 8),
    ax: "plt.Axes | None" = None,
    out_path: "str | None" = None,
):
    """
    Project the Q-space volume from
    :func:`~nrxrdct.laue.simulation.qspace_around_spot` onto the detector,
    so the same rod shown by :func:`plot_qspace_around_spot` can be compared
    directly against a real (or simulated) detector image.

    Every voxel with `vol['on_detector'] == True` is drawn at its actual
    `(xcam, ycam)` pixel position, using the same colour/size intensity
    encoding as :func:`plot_qspace_around_spot` (and the same
    normalisation, `I / I.max()` over the *whole* volume — pass matching
    `log_intensity` / `intensity_floor` to both calls) so the two plots read
    as two projections of one object: the 3-D view shows *why* a voxel is
    bright or excluded, this view shows *where* the surviving voxels
    actually land. Voxels that are reachable but off-detector, outside the
    energy window, or destructively cancelled to zero are simply absent
    here — exactly as they would be absent from a real exposure.

    The nominal reflection (`ΔQ = 0`, i.e. `hkl` itself) is marked with a
    crosshair so satellites can be read as offsets from it.

    Args:
        vol: Return value of
            :func:`~nrxrdct.laue.simulation.qspace_around_spot`, built
            **with a `camera`** (raises `ValueError` otherwise — there is
            no pixel information to plot without one).
        camera: Same `Camera` used to build `vol`. Used to draw the
            detector's pixel-area outline/limits; safe to omit if `image`
            is given (its shape is used instead) or if you just want the
            scatter auto-scaled to the data.
        image: Optional measured/simulated detector image (shape
            `(Nv, Nh)`), shown as a greyscale background so the projected
            rod can be matched against real spots.
        log_intensity, intensity_floor: Same meaning as in
            :func:`plot_qspace_around_spot`.
        figsize: Figure size in inches (only used when `ax is None`).
        ax: Draw into an existing 2-D :class:`~matplotlib.axes.Axes`;
            `None` creates a new figure.
        out_path: Save path; `None` → do not save.

    Returns:
        `(fig, ax)`

    Example:
    >>> vol = qspace_around_spot(stack, (0, 0, 2), camera=cam)
    >>> plot_qspace_around_spot(vol)                          # the rod, in Q
    >>> plot_qspace_on_detector(vol, camera=cam, image=img)   # same rod, on the chip
    """
    if vol.get("pix") is None:
        raise ValueError(
            "vol['pix'] is None — qspace_around_spot() must be called with "
            "camera=<Camera instance> to compute detector positions."
        )

    along = np.asarray(vol["along"])
    lat1 = np.asarray(vol["lateral1"])
    lat2 = np.asarray(vol["lateral2"])
    I = vol["I"]
    on_det = vol["on_detector"]
    pix = vol["pix"]

    pix_flat = pix.reshape(-1, 2)
    I_flat = I.reshape(-1)
    on_flat = on_det.reshape(-1)

    Imax = float(I.max()) if I.size and I.max() > 0 else 1.0
    if log_intensity:
        val_flat = np.log10(I_flat / Imax + intensity_floor)
        vmin, vmax = np.log10(intensity_floor), 0.0
    else:
        val_flat = I_flat / Imax
        vmin, vmax = 0.0, 1.0

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "qvol_warm", [_QVOL_LOW, _QVOL_MID, _QVOL_HIGH]
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    frac = np.clip((val_flat - vmin) / max(vmax - vmin, 1e-12), 0.0, 1.0)
    sizes = 6.0 + 90.0 * frac ** 1.5

    keep = on_flat
    xs, ys = pix_flat[keep, 0], pix_flat[keep, 1]
    vals = val_flat[keep]

    # ── nearest-to-G0 voxel (ΔQ = 0), for the crosshair ───────────────────────
    i0 = int(np.argmin(np.abs(along)))
    j0 = int(np.argmin(np.abs(lat1)))
    k0 = int(np.argmin(np.abs(lat2)))
    g0_pix = pix[i0, j0, k0] if on_det[i0, j0, k0] else None

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor(BG)
    else:
        fig = ax.figure

    ax.set_facecolor(BG)
    ax.tick_params(colors="#7788aa", labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1f2e")

    if image is not None:
        img = np.asarray(image, dtype=float)
        vmax_im = np.percentile(img[img > 0], 99) if img.max() > 0 else 1.0
        ax.imshow(
            np.log1p(img / vmax_im * 1000), origin="upper",
            cmap="gray", aspect="equal", interpolation="nearest",
        )
    elif camera is not None:
        ax.add_patch(Rectangle(
            (0, 0), camera.Nh, camera.Nv,
            linewidth=0.8, edgecolor="#333355", facecolor="none", zorder=0,
        ))

    sc = ax.scatter(
        xs, ys, c=vals, cmap=cmap, norm=norm, s=sizes[keep],
        marker="o", linewidths=0, alpha=0.9, zorder=3,
    )

    if g0_pix is not None:
        ax.scatter(
            [g0_pix[0]], [g0_pix[1]], s=140, marker="+",
            color=FG, linewidths=1.4, zorder=4,
        )

    if camera is not None:
        ax.set_xlim(0, camera.Nh)
        ax.set_ylim(camera.Nv, 0)
    elif len(xs):
        pad = 0.1 * max(xs.max() - xs.min(), ys.max() - ys.min(), 1.0)
        ax.set_xlim(xs.min() - pad, xs.max() + pad)
        ax.set_ylim(ys.max() + pad, ys.min() - pad)
    ax.set_aspect("equal")

    ax.set_xlabel("xcam  (px)", color=FG, fontsize=9)
    ax.set_ylabel("ycam  (px)", color=FG, fontsize=9)
    hkl = vol.get("hkl")
    layer_label = vol.get("layer")
    n_shown = int(keep.sum())
    n_total = int(on_flat.size)
    ax.set_title(
        f"Q-space rod projected onto detector — hkl={hkl}  (layer '{layer_label}')\n"
        f"{n_shown}/{n_total} voxels on-detector",
        color=FG, fontsize=9, pad=8,
    )

    cbar = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label("log₁₀ I / I_max" if log_intensity else "I / I_max", color=FG, fontsize=8)
    cbar.ax.yaxis.set_tick_params(color="#7788aa", labelsize=7)
    plt.setp(cbar.ax.get_yticklabels(), color=FG)
    cbar.outline.set_edgecolor("#333355")

    if standalone:
        fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {out_path}")

    return fig, ax


def plot_qspace_summary(
    vol: dict,
    *,
    per_layer_vols: "list[dict] | None" = None,
    camera: "Camera | None" = None,
    image: "np.ndarray | None" = None,
    log_intensity: bool = False,
    intensity_floor: float = 1e-6,
    resolution_dq: float = 0.01,
    figsize: tuple[float, float] = (18, 8),
    out_path: "str | None" = None,
):
    """
    Four-panel summary of the Q-space volume from
    :func:`~nrxrdct.laue.simulation.qspace_around_spot`.

    **Panel 1 — Rod profile** (top, full width): intensity averaged over the
    two transverse axes vs ΔQ along the rod direction.  A shaded band marks
    the accessible energy window; yellow dots mark voxels that reach the
    detector.  A secondary (right) y-axis overlays the photon energy E(ΔQ)
    along the rod centre, so fringe positions can be read directly in keV.

    **Panel 2 — Mid-plane heatmap** (bottom-left): ``pcolormesh`` of the
    volume slice at ΔQ_lateral₁ = 0 (i.e. ``I[:, n_lat//2, :]``), showing
    the rod shape and any lateral asymmetry in a legible 2-D image.  A
    contour marks the energy-window boundary.

    **Panel 3 — Detector projection** (bottom-centre): the same projection as
    :func:`plot_qspace_on_detector`, included only when ``vol['pix']`` is
    available.  Omitted (panel 2 widens) when no camera was used.

    **Panel 4 — Intensity comparison along the rod** (bottom-right): simulated
    intensity vs ΔQ_along for the primary ``vol``, restricted to on-detector
    voxels and normalised to its own peak.  When ``image`` is provided, the
    measured pixel intensity sampled along the same path is overlaid (also
    normalised).  When ``per_layer_vols`` is supplied (from
    :func:`~nrxrdct.laue.simulation.qspace_per_layer`), the x-axis switches
    to **absolute Q · n̂** so all layers' profiles can be compared on the same
    axis: each layer's on-detector 1D profile is drawn with a distinct colour,
    and the measured profile (if ``image`` is given) is drawn on top.

    Args:
        vol: Primary return value of
            :func:`~nrxrdct.laue.simulation.qspace_around_spot` — used for
            panels 1–3 and as the source of pixel positions for the measured
            intensity in panel 4.
        per_layer_vols: Optional list of vol dicts returned by
            :func:`~nrxrdct.laue.simulation.qspace_per_layer`.  When given,
            each layer's simulated 1D profile is overlaid in panel 4 on an
            absolute Q · n̂ axis so layers with different lattice parameters
            can be compared directly.
        camera: Same ``Camera`` used to build ``vol``; draws the detector
            boundary in the projection panel.
        image: Optional measured/simulated detector image (shape ``(Nv, Nh)``).
            Shown as a grey background in the projection panel, and sampled
            along the rod path for the intensity-comparison panel.
        log_intensity: Colour/scale by log₁₀(I) rather than raw I
            (default ``False``).
        intensity_floor: Floor added before the log (as a fraction of I_max).
            Only used when ``log_intensity=True``.
        figsize: Figure size in inches.
        out_path: Save path; ``None`` → do not save.

    Returns:
        ``(fig, axes)`` — axes is a list of 2, 3, or 4
        :class:`~matplotlib.axes.Axes` objects.

    Example:
    >>> vols = qspace_per_layer(stack, (0, 0, 2), camera=cam)
    >>> plot_qspace_summary(vols[0], per_layer_vols=vols, camera=cam,
    ...                     image=img, out_path="summary_002.png")
    """
    from scipy.ndimage import map_coordinates

    along = np.asarray(vol["along"])
    lat1 = np.asarray(vol["lateral1"])
    lat2 = np.asarray(vol["lateral2"])
    I = vol["I"]
    E = vol["E"]
    reachable = vol["reachable"]
    on_det = vol["on_detector"]
    has_pix = on_det is not None and vol.get("pix") is not None
    hkl = vol.get("hkl")
    layer_label = vol.get("layer")

    Imax = float(I.max()) if I.size and I.max() > 0 else 1.0
    if log_intensity:
        def _val(arr):
            return np.log10(np.asarray(arr) / Imax + intensity_floor)
        vlabel = "log₁₀ I / I_max"
        vmin, vmax_c = np.log10(intensity_floor), 0.0
    else:
        def _val(arr):
            return np.asarray(arr) / Imax
        vlabel = "I / I_max"
        vmin, vmax_c = 0.0, 1.0

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "qvol_warm", [_QVOL_LOW, _QVOL_MID, _QVOL_HIGH]
    )
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax_c)

    j0 = len(lat1) // 2
    k0 = len(lat2) // 2

    # ── figure / axes layout ──────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor(BG)
    if has_pix:
        gs = mgridspec.GridSpec(2, 3, figure=fig,
                                height_ratios=[1, 1.4], hspace=0.42, wspace=0.36)
        ax_rod = fig.add_subplot(gs[0, :])
        ax_heat = fig.add_subplot(gs[1, 0])
        ax_det = fig.add_subplot(gs[1, 1])
        ax_prof = fig.add_subplot(gs[1, 2])
        axes = [ax_rod, ax_heat, ax_det, ax_prof]
    else:
        gs = mgridspec.GridSpec(2, 1, figure=fig,
                                height_ratios=[1, 1.4], hspace=0.42)
        ax_rod = fig.add_subplot(gs[0])
        ax_heat = fig.add_subplot(gs[1])
        ax_det = None
        ax_prof = None
        axes = [ax_rod, ax_heat]

    # ══════════════════════════════════════════════════════════════════════════
    # Panel 1 — rod profile
    # ══════════════════════════════════════════════════════════════════════════
    _ax_style(ax_rod, f"Rod profile — hkl={hkl}  (layer '{layer_label}')")
    ax_rod.set_xlabel("ΔQ ∥ n̂  (Å⁻¹)", color=FG, fontsize=8)
    ax_rod.set_ylabel(vlabel, color=FG, fontsize=8)

    I_rod = I.mean(axis=(1, 2))
    val_rod = _val(I_rod)
    reach_rod = reachable.any(axis=(1, 2))

    if reach_rod.any():
        ax_rod.axvspan(
            along[reach_rod].min(), along[reach_rod].max(),
            color=_QVOL_WINDOW, alpha=0.10,
        )

    ax_rod.plot(along, val_rod, color=COL_SUP, lw=1.3, zorder=3)

    legend_h = []
    if has_pix:
        on_rod = on_det.any(axis=(1, 2))
        if on_rod.any():
            ax_rod.scatter(
                along[on_rod], val_rod[on_rod],
                s=20, color=COL_DB, zorder=4, linewidths=0,
            )
            legend_h.append(
                Line2D([0], [0], marker="o", linestyle="", color=COL_DB,
                       markersize=5, label="on detector")
            )
    if reach_rod.any():
        legend_h.append(
            Line2D([0], [0], color=_QVOL_WINDOW, lw=6, alpha=0.4,
                   label="energy window")
        )
    if legend_h:
        ax_rod.legend(handles=legend_h, fontsize=7, labelcolor=FG,
                      facecolor="#1a1f2e", edgecolor="#333355", loc="upper right")

    # right y-axis: photon energy along the rod centre
    E_rod = E[:, j0, k0]
    valid_E = np.isfinite(E_rod)
    if valid_E.sum() >= 2:
        ax_e = ax_rod.twinx()
        ax_e.set_facecolor(BG)
        ax_e.plot(along[valid_E], E_rod[valid_E] / 1000,
                  "--", color=COL_BCC, lw=0.9, alpha=0.75)
        ax_e.set_ylabel("E  (keV)", color=COL_BCC, fontsize=7, labelpad=4)
        ax_e.tick_params(colors=COL_BCC, labelsize=6)
        ax_e.yaxis.label.set_color(COL_BCC)
        for sp in ax_e.spines.values():
            sp.set_edgecolor("#1a1f2e")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel 2 — mid-plane heatmap  I(ΔQ_along, ΔQ_lat2)  at ΔQ_lat1 = 0
    # ══════════════════════════════════════════════════════════════════════════
    _ax_style(ax_heat, "Mid-plane heatmap  (ΔQ_lat₁ = 0)")
    ax_heat.set_xlabel("ΔQ ∥ n̂  (Å⁻¹)", color=FG, fontsize=8)
    ax_heat.set_ylabel("ΔQ transverse  (Å⁻¹)", color=FG, fontsize=8)

    slice2d = I[:, j0, :]                 # (n_along, n_lat2)
    heat_val = _val(slice2d).T            # (n_lat2, n_along) → rows = y, cols = x

    im = ax_heat.pcolormesh(
        along, lat2, heat_val, cmap=cmap, norm=norm, shading="nearest",
    )
    cbar_h = fig.colorbar(im, ax=ax_heat, shrink=0.9, pad=0.03)
    cbar_h.set_label(vlabel, color=FG, fontsize=7)
    cbar_h.ax.yaxis.set_tick_params(color="#7788aa", labelsize=6)
    plt.setp(cbar_h.ax.get_yticklabels(), color=FG)
    cbar_h.outline.set_edgecolor("#333355")

    reach_slice = reachable[:, j0, :]    # (n_along, n_lat2)
    if reach_slice.any() and not reach_slice.all():
        ax_heat.contour(
            along, lat2, reach_slice.T.astype(float),
            levels=[0.5], colors=[_QVOL_WINDOW], linewidths=0.8, alpha=0.8,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Panel 3 — detector projection.
    #
    # One marker per along step: coherent intensity |F_total|² and pixel
    # position are both averaged over all on-detector lateral (j,k) voxels
    # at that step, so each dot represents the full lateral footprint collapsed
    # to a single coherent-sum value.
    # ══════════════════════════════════════════════════════════════════════════
    if ax_det is not None:
        _ax_style(ax_det, "Detector projection")
        ax_det.set_xlabel("xcam  (px)", color=FG, fontsize=8)
        ax_det.set_ylabel("ycam  (px)", color=FG, fontsize=8)

        pix = vol["pix"]

        # collapse lateral: one (I, xcam, ycam) per along step
        n_on_lat_d = on_det.sum(axis=(1, 2))          # (n_along,)
        has_on = n_on_lat_d > 0
        denom = np.maximum(n_on_lat_d, 1)
        I_avg_d = np.where(has_on, (I * on_det).sum(axis=(1, 2)) / denom, np.nan)
        pix_x_avg = np.where(has_on, (pix[..., 0] * on_det).sum(axis=(1, 2)) / denom, np.nan)
        pix_y_avg = np.where(has_on, (pix[..., 1] * on_det).sum(axis=(1, 2)) / denom, np.nan)

        xs_d = pix_x_avg[has_on]
        ys_d = pix_y_avg[has_on]
        val_d = _val(I_avg_d[has_on])
        frac_d = np.clip((val_d - vmin) / max(vmax_c - vmin, 1e-12), 0.0, 1.0)
        sizes_d = 8.0 + 100.0 * frac_d ** 1.5

        if image is not None:
            img_arr = np.asarray(image, dtype=float)
            vmax_im = np.percentile(img_arr[img_arr > 0], 99) if img_arr.max() > 0 else 1.0
            ax_det.imshow(
                np.log1p(img_arr / vmax_im * 1000), origin="upper",
                cmap="gray", aspect="equal", interpolation="nearest",
            )
        if camera is not None:
            ec = "#6688bb" if image is not None else "#333355"
            ax_det.add_patch(Rectangle(
                (0, 0), camera.Nh, camera.Nv,
                linewidth=0.8, edgecolor=ec, facecolor="none", zorder=5,
            ))

        rgba_d = cmap(norm(val_d)).copy()
        rgba_d[:, 3] = np.clip(0.15 + 0.8 * frac_d, 0.15, 0.95)
        sc_d = ax_det.scatter(
            xs_d, ys_d, facecolors=rgba_d,
            s=sizes_d, marker="o", linewidths=0, zorder=3,
        )

        # crosshair at G0 (ΔQ = 0)
        i0 = int(np.argmin(np.abs(along)))
        if has_on[i0]:
            ax_det.scatter(
                [pix_x_avg[i0]], [pix_y_avg[i0]], s=120, marker="+",
                color=FG, linewidths=1.2, zorder=4,
            )

        if camera is not None:
            ax_det.set_xlim(0, camera.Nh)
            ax_det.set_ylim(camera.Nv, 0)
        elif len(xs_d):
            pad = 0.1 * max(xs_d.max() - xs_d.min(), ys_d.max() - ys_d.min(), 1.0)
            ax_det.set_xlim(xs_d.min() - pad, xs_d.max() + pad)
            ax_det.set_ylim(ys_d.max() + pad, ys_d.min() - pad)
        ax_det.set_aspect("equal")

        _sm_d = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        _sm_d.set_array([])
        cbar_d = fig.colorbar(_sm_d, ax=ax_det, shrink=0.8, pad=0.03)
        cbar_d.set_label(vlabel, color=FG, fontsize=7)
        cbar_d.ax.yaxis.set_tick_params(color="#7788aa", labelsize=6)
        plt.setp(cbar_d.ax.get_yticklabels(), color=FG)
        cbar_d.outline.set_edgecolor("#333355")

    # ══════════════════════════════════════════════════════════════════════════
    # Panel 4 — coherent total intensity along the on-detector rod path.
    #
    # vol["I"] = |F_total|²  where  F_total = Σ_layer F_layer·exp(iφ_layer)
    # is the coherent structure-factor sum already computed by
    # structure_factor_batch — phase shifts between layers are included.
    # Intensities are averaged over ALL on-detector (j,k) lateral voxels at
    # each along step, then normalised to their on-detector maximum.
    # ══════════════════════════════════════════════════════════════════════════
    if ax_prof is not None:
        _ax_style(ax_prof, "Coherent intensity along detector projection")
        ax_prof.set_xlabel("ΔQ ∥ n̂  (Å⁻¹)", color=FG, fontsize=8)
        ax_prof.set_ylabel("I / I_max", color=FG, fontsize=8)
        ax_prof.grid(True, ls=":", lw=0.35, color="#181e2e")

        n_on_lat = on_det.sum(axis=(1, 2))          # (n_along,)
        has_any = n_on_lat > 0
        I_sum_lat = (I * on_det).sum(axis=(1, 2))
        I_sim_avg = np.where(has_any, I_sum_lat / np.maximum(n_on_lat, 1), np.nan)
        along_on = along[has_any]
        I_sim_on = I_sim_avg[has_any]
        peak_sim = float(np.nanmax(I_sim_on)) if np.any(np.isfinite(I_sim_on)) else 1.0
        I_sim_norm = I_sim_on / max(peak_sim, 1e-30)

        prof_handles = []
        if has_any.any():
            from scipy.ndimage import gaussian_filter1d as _gf1d
            dq_step = float(np.diff(along_on).mean()) if len(along_on) > 1 else 1.0
            sigma_px = max(resolution_dq / abs(dq_step), 0.5)
            _env = _gf1d(I_sim_norm, sigma=sigma_px)
            _env_label = f"convolved (σ={resolution_dq*1e3:.1f} mÅ⁻¹)"

            if image is not None and has_pix:
                img_arr = np.asarray(image, dtype=float)
                pix_v = vol["pix"]
                on_flat_idx = np.where(on_det.reshape(len(along), -1))
                i_idx, jk_flat = on_flat_idx
                n_lat2 = on_det.shape[2]
                j_idx = jk_flat // n_lat2
                k_idx = jk_flat % n_lat2
                pix_sel = pix_v[i_idx, j_idx, k_idx]
                coords = np.array([pix_sel[:, 1], pix_sel[:, 0]])
                meas_vals = map_coordinates(img_arr, coords, order=1, mode="nearest")
                I_meas_sum = np.zeros(len(along))
                np.add.at(I_meas_sum, i_idx, meas_vals)
                I_meas_avg = np.where(has_any, I_meas_sum / np.maximum(n_on_lat, 1), np.nan)
                I_meas_on = I_meas_avg[has_any]
                peak_meas = float(np.nanmax(I_meas_on)) if np.any(np.isfinite(I_meas_on)) else 1.0
                ax_prof.plot(along_on, I_meas_on / max(peak_meas, 1e-30),
                             color=FG, lw=1.3, zorder=3)
                ax_prof.plot(along_on, I_sim_norm, color=COL_SUP, lw=1.3,
                             ls="--", zorder=4)
                ax_prof.plot(along_on, _env, color=COL_BCC, lw=1.5,
                             ls="-", alpha=0.75, zorder=5)
                prof_handles = [
                    Line2D([0], [0], color=FG, lw=1.3, label="measured"),
                    Line2D([0], [0], color=COL_SUP, lw=1.3, ls="--",
                           label="simulated (coherent)"),
                    Line2D([0], [0], color=COL_BCC, lw=1.5,
                           alpha=0.75, label=_env_label),
                ]
            else:
                ax_prof.plot(along_on, I_sim_norm, color=COL_SUP, lw=1.3, zorder=3)
                ax_prof.plot(along_on, _env, color=COL_BCC, lw=1.5,
                             ls="-", alpha=0.75, zorder=4)
                prof_handles = [
                    Line2D([0], [0], color=COL_SUP, lw=1.3,
                           label="simulated (coherent)"),
                    Line2D([0], [0], color=COL_BCC, lw=1.5,
                           alpha=0.75, label=_env_label),
                ]
            ax_prof.set_ylim(bottom=0)
            ax_prof.legend(
                handles=prof_handles, fontsize=7, labelcolor=FG,
                facecolor="#1a1f2e", edgecolor="#333355",
                loc="upper left", bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0, handlelength=1.4,
            )
        else:
            ax_prof.text(0.5, 0.5, "no on-detector voxels",
                         color="#7788aa", ha="center", va="center",
                         transform=ax_prof.transAxes, fontsize=8)

    fig.suptitle(
        f"Q-space summary — hkl={hkl}  (layer '{layer_label}')",
        color=FG, fontsize=11, y=1.01,
    )

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {out_path}")

    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATED DETECTOR IMAGE
# ─────────────────────────────────────────────────────────────────────────────


def plot_detector_projection(
    vol_or_vols,
    *,
    image: "np.ndarray | None" = None,
    camera: "Camera | None" = None,
    pad_px: int = 20,
    log_intensity: bool = True,
    psf_sigma_px: float = 1.0,
    exclude_bragg_along: "float | None" = None,
    vmin_percentile: float = 0.0,
    vmax_percentile: float = 99.5,
    figsize: "tuple[float, float] | None" = None,
    out_path: "str | None" = None,
) -> tuple:
    """
    Plot the Q-space projection onto detector pixels produced by
    :func:`~nrxrdct.laue.simulation.project_to_detector`.

    When *image* (a full measured detector frame) is supplied the function
    shows two panels side by side — simulated (left) and measured patch
    (right) — so you can compare them directly.

    **Normalisation**

    Each panel is independently normalised by ``I/I_max`` before the
    log-stretch: the simulated panel divides by its own peak intensity, the
    measured panel divides by the measured intensity at the simulated spot
    positions (or by its own peak if none are non-zero).  A shared
    percentile-based colour scale (*vmin_percentile* / *vmax_percentile*)
    is then derived from the combined non-zero pixels of both stretched
    panels so that faint features (e.g. superlattice satellites) are
    visible at the same visual level.

    **Auto-crop**

    After the PSF convolution the function crops the arrays to the
    bounding box of non-zero simulated pixels, expanded by *pad_px* on
    every side.  This is done even when a full-frame *camera* is supplied
    (where :func:`~nrxrdct.laue.simulation.project_to_detector` returns a
    ~2k × 2k frame) so that a 1–2 pixel spot is never lost in an empty
    field.

    Args:
        vol_or_vols: A single vol dict from ``qspace_around_spot`` or a list
            of such dicts (e.g. from ``qspace_per_layer``).  Multiple volumes
            are summed onto the same pixel grid.
        image: Full measured detector frame (``ndarray``, shape ``(Nv, Nh)``).
            When given a second panel is added showing the corresponding patch.
        camera: :class:`~nrxrdct.laue.camera.Camera`.  If ``None`` the output
            is cropped to the spot bounding box (+ *pad_px*).
        pad_px: Extra margin (pixels) around the bounding box patch.
        log_intensity: Apply ``log1p`` stretch to both panels.
        psf_sigma_px: Gaussian PSF sigma in pixels applied to the simulated
            image before normalisation.  Models detector point-spread (charge
            sharing, optical blur).  Set to 0 to disable.
        exclude_bragg_along: float or None.  When set, voxels with
            ``|along| < exclude_bragg_along`` (Å⁻¹) are stripped from the
            simulated projection so that the kinematical Bragg peak does not
            flood the image.  A good starting value is ``π / Λ`` (e.g. 0.03
            for Λ=100 Å).
        vmin_percentile: Percentile (0–100) of the combined non-zero pixels
            from both panels used as the colour-scale minimum.  Default 0
            (true minimum — no black-level clipping).  Increase to suppress
            faint background; e.g. 5 clips the bottom 5 % of signal.
        vmax_percentile: Percentile (0–100) used as the colour-scale maximum.
            Default 99.5 — clips the top 0.5 % so a single bright outlier
            (e.g. unsuppressed Bragg peak) does not compress the satellite
            colour range.  Set to 100 for the true maximum.
        figsize: Figure size.  Defaults to ``(8, 5)`` (one panel) or
            ``(14, 5)`` (two panels).
        out_path: If given, save the figure to this path.

    Returns:
        ``(fig, axes)`` where *axes* is a 1- or 2-element list.
    """
    from .simulation import project_to_detector

    sim, x0, y0 = project_to_detector(
        vol_or_vols, pad_px=pad_px, camera=camera,
        exclude_bragg_along=exclude_bragg_along,
    )
    Ny, Nx = sim.shape

    # ── PSF convolution (before normalisation so it acts on raw flux) ────────
    if psf_sigma_px > 0:
        from scipy.ndimage import gaussian_filter as _gf
        sim = _gf(sim, sigma=psf_sigma_px)

    # ── crop to bounding box of non-zero pixels ───────────────────────────────
    # project_to_detector already crops when camera=None.  When camera is
    # supplied it returns the full detector frame, making any spot invisible.
    # Crop here so the display always focuses on the simulated region.
    nz = np.argwhere(sim > 0)
    if len(nz):
        ry0, rx0 = nz[:, 0].min(), nz[:, 1].min()
        ry1, rx1 = nz[:, 0].max() + 1, nz[:, 1].max() + 1
        ry0 = max(ry0 - pad_px, 0)
        rx0 = max(rx0 - pad_px, 0)
        ry1 = min(ry1 + pad_px, Ny)
        rx1 = min(rx1 + pad_px, Nx)
        sim = sim[ry0:ry1, rx0:rx1]
        x0, y0 = x0 + rx0, y0 + ry0
        Ny, Nx = sim.shape

    # pixel-space extent for imshow: (left, right, bottom, top) with origin='upper'
    ext = [x0, x0 + Nx, y0 + Ny, y0]

    has_meas = image is not None
    n_panels = 2 if has_meas else 1
    if figsize is None:
        figsize = (14.0, 5.5) if has_meas else (7.0, 5.5)

    fig, axes_arr = plt.subplots(
        1, n_panels, figsize=figsize,
        facecolor=BG, squeeze=False,
        sharex=True, sharey=True,
    )
    axes = list(axes_arr[0])
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "qvol_warm", [_QVOL_LOW, _QVOL_MID, _QVOL_HIGH]
    )

    def _stretch(arr):
        if log_intensity:
            return np.log1p(arr)
        return arr.copy()

    vols_list = [vol_or_vols] if isinstance(vol_or_vols, dict) else list(vol_or_vols)

    # ── I / I_max normalisation ───────────────────────────────────────────────
    # Simulated panel: divide by its own maximum so the brightest simulated
    # feature (satellite peak with exclude_bragg_along set, or Bragg peak
    # otherwise) maps to 1.0.
    I_max_sim = float(sim.max()) if sim.max() > 0 else 1.0
    sim = sim / I_max_sim

    if has_meas:
        img_arr = np.asarray(image, dtype=float)
        meas_patch = img_arr[y0:y0 + Ny, x0:x0 + Nx]
        I_max_meas = float(meas_patch.max()) if meas_patch.max() > 0 else 1.0
        meas_norm = meas_patch / I_max_meas

    sim_s = _stretch(sim)

    # ── colour-scale limits from percentiles of non-zero pixels ───────────────
    # Collect stretched values from both panels (non-zero only so the large
    # empty background does not pull the percentiles to zero).
    _nz_vals = [sim_s[sim_s > 0]]
    if has_meas:
        meas_s_tmp = _stretch(meas_norm)
        _nz_m = meas_s_tmp[meas_s_tmp > 0]
        if len(_nz_m):
            _nz_vals.append(_nz_m)
    _all_nz = np.concatenate(_nz_vals) if _nz_vals else np.array([0.0, 1.0])
    vmin_s = float(np.percentile(_all_nz, vmin_percentile)) if len(_all_nz) else 0.0
    vmax_s = float(np.percentile(_all_nz, vmax_percentile)) if len(_all_nz) else 1.0
    if vmax_s <= vmin_s:
        vmax_s = vmin_s + 1e-6

    for ax in axes:
        _ax_style(ax, "")
        ax.set_xlabel("xcam  (px)", color=FG, fontsize=8)
        ax.set_ylabel("ycam  (px)", color=FG, fontsize=8)

    # ── left panel: simulated ─────────────────────────────────────────────────
    ax_sim = axes[0]
    _sim_title = "Simulated"
    ax_sim.set_title(_sim_title, color=FG, fontsize=9, pad=4)
    im_s = ax_sim.imshow(
        sim_s, origin="upper", extent=ext,
        cmap=cmap, vmin=vmin_s, vmax=vmax_s,
        aspect="equal", interpolation="nearest",
    )
    cbar_s = fig.colorbar(im_s, ax=ax_sim, shrink=0.85, pad=0.03)
    cbar_s.set_label("log(1+counts)" if log_intensity else "counts", color=FG, fontsize=7)
    cbar_s.ax.yaxis.set_tick_params(color="#7788aa", labelsize=6)
    plt.setp(cbar_s.ax.get_yticklabels(), color=FG)
    cbar_s.outline.set_edgecolor("#333355")

    # ── right panel: measured patch ───────────────────────────────────────────
    if has_meas:
        ax_meas = axes[1]
        ax_meas.set_title("Measured", color=FG, fontsize=9, pad=4)
        meas_s = meas_s_tmp
        im_m = ax_meas.imshow(
            meas_s, origin="upper", extent=ext,
            cmap="gray", vmin=vmin_s, vmax=vmax_s,
            aspect="equal", interpolation="nearest",
        )
        cbar_m = fig.colorbar(im_m, ax=ax_meas, shrink=0.85, pad=0.03)
        cbar_m.set_label("log(1+counts)" if log_intensity else "counts", color=FG, fontsize=7)
        cbar_m.ax.yaxis.set_tick_params(color="#7788aa", labelsize=6)
        plt.setp(cbar_m.ax.get_yticklabels(), color=FG)
        cbar_m.outline.set_edgecolor("#333355")

    # ── crosshair at nominal Bragg pixel (along=0, lateral centre) ───────────
    # Draw one + per unique HKL using the first vol for that HKL that has a
    # finite (on-detector) G0 pixel.  Using vols_list[0] blindly fails when
    # the first vol is a substrate layer whose G0 is off-detector.
    seen_hkls: set = set()
    for _vol in vols_list:
        _hkl = _vol.get("hkl")
        if _hkl in seen_hkls:
            continue
        _pix = _vol.get("pix")
        _along = _vol.get("along")
        if _pix is None or _along is None:
            continue
        _i0 = int(np.argmin(np.abs(_along)))
        _j0 = _pix.shape[1] // 2
        _k0 = _pix.shape[2] // 2
        _cx = float(_pix[_i0, _j0, _k0, 0])
        _cy = float(_pix[_i0, _j0, _k0, 1])
        if np.isfinite(_cx) and np.isfinite(_cy):
            for ax in axes:
                ax.plot(_cx, _cy, "+", color=FG, ms=10, mew=1.2, zorder=5)
            seen_hkls.add(_hkl)

    # ── build title ───────────────────────────────────────────────────────────
    unique_hkls = list(dict.fromkeys(str(v.get("hkl", "?")) for v in vols_list))
    unique_layers = list(dict.fromkeys(str(v.get("layer", "?")) for v in vols_list))
    hkl_str = unique_hkls[0] if len(unique_hkls) == 1 else f"{len(unique_hkls)} reflections"
    layer_str = unique_layers[0] if len(unique_layers) == 1 else f"{len(unique_layers)} layers"
    title = f"Detector projection — hkl={hkl_str}  ({layer_str})"

    fig.suptitle(title, color=FG, fontsize=10, y=1.01)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved → {out_path}")

    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# PER-PIXEL SPOT IMAGE: SIMULATED vs MEASURED
# ─────────────────────────────────────────────────────────────────────────────


def plot_spot_image(
    img,
    *,
    exp_image=None,
    log_intensity=True,
    vmin_percentile=0.0,
    vmax_percentile=99.5,
    figsize=None,
    out_path=None,
):
    """
    Plot the output of :func:`~nrxrdct.laue.simulation.simulate_spot_image`,
    optionally next to the matching patch cropped from a real measured
    detector frame.

    **Normalisation.** Each panel is independently normalised by its own
    `I / I_max` before the log-stretch — the simulated panel by its own
    peak, the measured patch by its own peak within the cropped window.
    A shared percentile-based colour scale (`vmin_percentile` /
    `vmax_percentile`) is then derived from the combined non-zero pixels of
    both stretched panels, so faint features are visible at the same
    visual level in both — same convention as
    :func:`plot_detector_projection`.

    Args:
        img (dict): Return value of
            :func:`~nrxrdct.laue.simulation.simulate_spot_image`.
        exp_image (ndarray, shape (Nv, Nh), optional): Full measured
            detector frame, in the same pixel convention as the `camera`
            used to build `img`.  When given, a second "Measured" panel is
            added, cropped to `img['x0']` / `img['y0']`.
        log_intensity (bool): Apply `log1p` stretch to both panels.
        vmin_percentile, vmax_percentile (float): Percentile (0-100) of the
            combined non-zero stretched pixels used as the colour-scale
            min/max.  Defaults `0` / `99.5` — true minimum, top 0.5% clipped.
        figsize: Figure size.  Defaults to `(7, 5.5)` (one panel) or
            `(14, 5.5)` (two panels).
        out_path (str or None): Save figure to this path if given.

    Returns:
        ``(fig, axes)`` where *axes* is a 1- or 2-element list.
    """
    sim = np.asarray(img["I"], dtype=float)
    x0, y0 = img["x0"], img["y0"]
    ext = [x0[0] - 0.5, x0[-1] + 0.5, y0[-1] + 0.5, y0[0] - 0.5]

    has_meas = exp_image is not None
    n_panels = 2 if has_meas else 1
    if figsize is None:
        figsize = (14.0, 5.5) if has_meas else (7.0, 5.5)

    fig, axes_arr = plt.subplots(
        1, n_panels, figsize=figsize, facecolor=BG, squeeze=False,
        sharex=True, sharey=True,
    )
    axes = list(axes_arr[0])
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "qvol_warm", [_QVOL_LOW, _QVOL_MID, _QVOL_HIGH]
    )

    def _stretch(arr):
        return np.log1p(arr) if log_intensity else arr.copy()

    I_max_sim = float(sim.max()) if sim.max() > 0 else 1.0
    sim_n = sim / I_max_sim
    sim_s = _stretch(sim_n)

    if has_meas:
        img_arr = np.asarray(exp_image, dtype=float)
        iy0, iy1 = int(round(y0[0])), int(round(y0[-1])) + 1
        ix0, ix1 = int(round(x0[0])), int(round(x0[-1])) + 1
        meas_patch = img_arr[iy0:iy1, ix0:ix1]
        I_max_meas = float(meas_patch.max()) if meas_patch.max() > 0 else 1.0
        meas_n = meas_patch / I_max_meas
        meas_s = _stretch(meas_n)

    _nz_vals = [sim_s[sim_s > 0]]
    if has_meas:
        _nz_m = meas_s[meas_s > 0]
        if len(_nz_m):
            _nz_vals.append(_nz_m)
    _all_nz = np.concatenate(_nz_vals) if _nz_vals else np.array([0.0, 1.0])
    vmin_s = float(np.percentile(_all_nz, vmin_percentile)) if len(_all_nz) else 0.0
    vmax_s = float(np.percentile(_all_nz, vmax_percentile)) if len(_all_nz) else 1.0
    if vmax_s <= vmin_s:
        vmax_s = vmin_s + 1e-6

    for ax in axes:
        _ax_style(ax, "")
        ax.set_xlabel("xcam  (px)", color=FG, fontsize=8)
        ax.set_ylabel("ycam  (px)", color=FG, fontsize=8)

    ax_sim = axes[0]
    ax_sim.set_title("Simulated", color=FG, fontsize=9, pad=4)
    im_s = ax_sim.imshow(
        sim_s, origin="upper", extent=ext, cmap=cmap,
        vmin=vmin_s, vmax=vmax_s, aspect="equal", interpolation="nearest",
    )
    cbar_s = fig.colorbar(im_s, ax=ax_sim, shrink=0.85, pad=0.03)
    cbar_s.set_label("log(1+I/I_max)" if log_intensity else "I/I_max", color=FG, fontsize=7)
    cbar_s.ax.yaxis.set_tick_params(color="#7788aa", labelsize=6)
    plt.setp(cbar_s.ax.get_yticklabels(), color=FG)
    cbar_s.outline.set_edgecolor("#333355")

    if has_meas:
        ax_meas = axes[1]
        ax_meas.set_title("Measured", color=FG, fontsize=9, pad=4)
        im_m = ax_meas.imshow(
            meas_s, origin="upper", extent=ext, cmap="gray",
            vmin=vmin_s, vmax=vmax_s, aspect="equal", interpolation="nearest",
        )
        cbar_m = fig.colorbar(im_m, ax=ax_meas, shrink=0.85, pad=0.03)
        cbar_m.set_label("log(1+I/I_max)" if log_intensity else "I/I_max", color=FG, fontsize=7)
        cbar_m.ax.yaxis.set_tick_params(color="#7788aa", labelsize=6)
        plt.setp(cbar_m.ax.get_yticklabels(), color=FG)
        cbar_m.outline.set_edgecolor("#333355")

    ref_x, ref_y = img.get("ref_pix", (None, None))
    if ref_x is not None:
        for ax in axes:
            ax.plot(ref_x, ref_y, "+", color=FG, ms=10, mew=1.2, zorder=5)

    hkl_str = str(img.get("hkl", "?"))
    layer_str = str(img.get("layer", "?"))
    E0 = img.get("E0")
    n_harm = img.get("n_harmonics")
    title = f"hkl={hkl_str}  ({layer_str})  E0={E0:.0f} eV  harmonics=1..{n_harm}"
    fig.suptitle(title, color=FG, fontsize=10, y=1.01)
    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved → {out_path}")

    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# PIXEL DEVIATION: SIMULATED vs MEASURED
# ─────────────────────────────────────────────────────────────────────────────


def plot_pix_deviation(
    spots: list[dict],
    peaklist: "np.ndarray",
    *,
    max_dist_px: float = 15.0,
    properties: list[str] | None = None,
    figsize: tuple[float, float] = (14, 10),
    out_path: "str | None" = None,
) -> tuple:
    """
    Match simulated spots to measured peaks and plot pixel deviations
    as a function of spot properties.

    Each simulated spot is paired with the nearest measured peak within
    *max_dist_px* pixels (Euclidean distance in detector pixel space).
    Unmatched spots are excluded.  The deviation is defined as:

        Δcol = sim_col − meas_col
        Δrow = sim_row − meas_row

    Args:
        spots: output of any ``simulate_laue_*`` function.
        peaklist: ``(N, ≥2)`` array with columns ``[col, row, ...]``
            (same format as :func:`convert_spotsfile2peaklist`).
        max_dist_px: matching radius in pixels.
        properties: spot dict keys to plot against deviations.
            Default: ``['E', 'tth', 'chi', 'intensity']``.
            Any key present in the spot dicts is valid (e.g.
            ``'az'``, ``'lambda'``, ``'source_depth_mm'``).
        figsize: figure size.
        out_path: save path; ``None`` → do not save.

    Returns:
        ``(fig, axes, matched)`` where *matched* is a list of dicts with
        keys ``'spot'``, ``'peak_idx'``, ``'meas_col'``, ``'meas_row'``,
        ``'dx'``, ``'dy'``, ``'dist'``, ``'phase'``.
    """
    from scipy.spatial import cKDTree

    if properties is None:
        properties = ["E", "tth", "chi", "intensity"]

    pl = np.asarray(peaklist, dtype=float)
    meas_xy = pl[:, :2]
    tree = cKDTree(meas_xy)

    matched: list[dict] = []
    for spot in spots:
        pix = spot.get("pix")
        if pix is None:
            continue
        sim_col, sim_row = float(pix[0]), float(pix[1])
        dist, idx = tree.query([sim_col, sim_row], k=1)
        if dist <= max_dist_px:
            matched.append({
                "spot":      spot,
                "peak_idx":  int(idx),
                "meas_col":  float(meas_xy[idx, 0]),
                "meas_row":  float(meas_xy[idx, 1]),
                "dx":        sim_col - float(meas_xy[idx, 0]),
                "dy":        sim_row - float(meas_xy[idx, 1]),
                "dist":      float(dist),
                "phase":     spot.get("phase_label", "unknown"),
            })

    if not matched:
        raise ValueError(
            f"No spots matched within {max_dist_px} px. "
            "Try increasing max_dist_px."
        )

    phases = list(dict.fromkeys(m["phase"] for m in matched))
    palette = plt.get_cmap("tab10")
    phase_color = {ph: palette(i % 10) for i, ph in enumerate(phases)}

    dx_all = np.array([m["dx"] for m in matched])
    dy_all = np.array([m["dy"] for m in matched])
    dist_all = np.sqrt(dx_all ** 2 + dy_all ** 2)
    med = float(np.median(dist_all))
    rms = float(np.sqrt(np.mean(dist_all ** 2)))

    # ── Layout: row 0 = 2D scatter + histogram; rows 1+ = 2 props per row ────
    n_props = len(properties)
    n_prop_rows = (n_props + 1) // 2   # ceil(n_props / 2)
    n_rows = 1 + n_prop_rows
    fig, axes = plt.subplots(n_rows, 2, figsize=figsize,
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    if n_rows == 1:
        axes = axes[np.newaxis, :]   # ensure 2D
    fig.patch.set_facecolor(BG)
    for ax in axes.flat:
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a1f2e")

    # ── Row 0 left: 2D (Δcol, Δrow) scatter ──────────────────────────────────
    ax_2d = axes[0, 0]
    for ph in phases:
        ms = [m for m in matched if m["phase"] == ph]
        ax_2d.scatter(
            [m["dx"] for m in ms], [m["dy"] for m in ms],
            s=14, color=phase_color[ph], alpha=0.75, label=ph, zorder=3,
            edgecolors="none",
        )
    ax_2d.axhline(0, color=FG, lw=0.5, alpha=0.4)
    ax_2d.axvline(0, color=FG, lw=0.5, alpha=0.4)
    ax_2d.set_xlabel("Δcol  (px)", color=FG, fontsize=8)
    ax_2d.set_ylabel("Δrow  (px)", color=FG, fontsize=8)
    ax_2d.set_title("Sim − Meas  (2D)", color=FG, fontsize=8)
    ax_2d.set_aspect("equal")
    if len(phases) > 1:
        ax_2d.legend(fontsize=7, labelcolor=FG,
                     facecolor="#1a1f2e", edgecolor="#333355")

    # ── Row 0 right: |Δpix| histogram ────────────────────────────────────────
    ax_hist = axes[0, 1]
    ax_hist.hist(dist_all, bins=min(40, max(len(matched) // 2, 5)),
                 color="#4488cc", alpha=0.85, edgecolor="none")
    ax_hist.axvline(med, color="orange", lw=1, label=f"median {med:.2f} px")
    ax_hist.axvline(rms, color="#ee4444", lw=1, linestyle="--",
                    label=f"RMS {rms:.2f} px")
    ax_hist.set_xlabel("|Δpix|  (px)", color=FG, fontsize=8)
    ax_hist.set_ylabel("count", color=FG, fontsize=8)
    ax_hist.set_title(f"|Δpix| distribution  ({len(matched)} matched)",
                      color=FG, fontsize=8)
    ax_hist.legend(fontsize=7, labelcolor=FG,
                   facecolor="#1a1f2e", edgecolor="#333355")

    # ── Rows 1+: |Δpix| vs each property, 2 per row ──────────────────────────
    _labels = {
        "E":               "E  (keV)",
        "tth":             "2θ  (°)",
        "chi":             "χ  (°)",
        "az":              "az  (°)",
        "intensity":       "intensity  (norm.)",
        "lambda":          "λ  (Å)",
        "source_depth_mm": "depth  (mm)",
        "F2":              "F²",
    }
    _scale = {"E": 1e-3}

    for i, prop in enumerate(properties):
        row = 1 + i // 2
        col = i % 2
        ax = axes[row, col]
        xlabel = _labels.get(prop, prop)
        scale = _scale.get(prop, 1.0)

        for ph in phases:
            ms = [m for m in matched if m["phase"] == ph]
            vals = [m["spot"].get(prop, np.nan) * scale for m in ms]
            dists = [m["dist"] for m in ms]
            ax.scatter(vals, dists, s=10, color=phase_color[ph], alpha=0.7,
                       edgecolors="none", zorder=3, label=ph)

        ax.set_xlabel(xlabel, color=FG, fontsize=8)
        ax.set_ylabel("|Δpix|  (px)", color=FG, fontsize=8)
        if len(phases) > 1:
            ax.legend(fontsize=6, labelcolor=FG,
                      facecolor="#1a1f2e", edgecolor="#333355")

    # hide unused axes if n_props is odd
    if n_props % 2 == 1:
        axes[-1, -1].set_visible(False)

    fig.suptitle(
        f"Pixel deviation: simulation vs measurement  "
        f"({len(matched)} matched, max_dist={max_dist_px:.0f} px  |  "
        f"RMS={rms:.2f} px  |  median={med:.2f} px)",
        color=FG, fontsize=9, y=1.01,
    )

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())

    return fig, axes, matched


def plot_depth_scan_image(
    result: dict,
    stack: "LayeredCrystal | None" = None,
    *,
    top_n_spots: int = 20,
    figsize: tuple[float, float] = (12, 8),
    cmap_matrix: str = "inferno",
    out_path: "str | None" = None,
    ax_score: "plt.Axes | None" = None,
) -> tuple:
    """
    Visualise the output of :func:`~nrxrdct.laue.simulation.depth_scan_image`.

    Three panels are drawn:

    * **Left** — Global depth score profile (total + per-phase).
    * **Centre** — Score matrix heat-map: rows are depth steps, columns are the
      ``top_n_spots`` most responsive spots (sorted by peak score).  The
      colour scale is per-column so dim spots are not swamped by bright ones.
    * **Right** (optional inset) — Score matrix for *all* valid spots so that
      the column ordering can be inspected.

    Args:
        result: Dict returned by :func:`depth_scan_image`.
        top_n_spots: Number of highest-scoring spot columns to show in the
            centre heatmap.
        figsize: Figure size ``(width, height)`` in inches.
        cmap_matrix: Colormap name for the score matrix.
        out_path: If given, save the figure to this path.
        ax_score: Optional pre-existing :class:`~matplotlib.axes.Axes` for the
            score profile.  If supplied the figure is *not* created internally
            and only the score axes is drawn (useful for inline notebooks).

    Returns:
        ``(fig, axes)`` where *axes* is a list
        ``[ax_score, ax_heatmap]`` (``ax_heatmap`` is ``None`` when
        *ax_score* was supplied externally).
    """
    z_mm         = result["z_mm"]
    score        = result["score"]
    score_phase  = result["score_per_phase"]
    score_matrix = result["score_matrix"]   # (n_steps, n_valid)
    phases       = list(score_phase.keys())

    PHASE_COLS = ["#4fc3f7", "#ff6633", "#ffffaa", "#aaffaa", "#ff88ff"]
    phase_color = {ph: PHASE_COLS[i % len(PHASE_COLS)] for i, ph in enumerate(phases)}

    # ── Figure layout ─────────────────────────────────────────────────────────
    if ax_score is not None:
        fig = ax_score.get_figure()
        ax_s = ax_score
        ax_h = None
        axes_out = [ax_s, ax_h]
    else:
        fig, (ax_s, ax_h) = plt.subplots(
            1, 2,
            figsize=figsize,
            gridspec_kw={"width_ratios": [1, 2]},
            facecolor=BG,
        )
        axes_out = [ax_s, ax_h]

    for ax in [ax_s] + ([ax_h] if ax_h is not None else []):
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a1f2e")

    # ── Score profile (left panel) ────────────────────────────────────────────
    ax_s.plot(z_mm * 1e3, score, color=FG, lw=1.5, label="total", zorder=5)
    for ph in phases:
        ax_s.plot(z_mm * 1e3, score_phase[ph], lw=1, color=phase_color[ph],
                  linestyle="--", label=ph, alpha=0.8)

    z_best = float(z_mm[np.argmax(score)])
    ax_s.axvline(z_best * 1e3, color="orange", lw=1, linestyle=":",
                 label=f"peak z={z_best*1e3:.1f} µm")
    ax_s.set_xlabel("depth  (µm)", color=FG, fontsize=8)
    ax_s.set_ylabel("image score  (a.u.)", color=FG, fontsize=8)
    ax_s.set_title("Depth score profile", color=FG, fontsize=9)
    ax_s.legend(fontsize=7, labelcolor=FG, facecolor="#1a1f2e", edgecolor="#333355")

    # ── Layer interfaces ──────────────────────────────────────────────────────
    if stack is not None:
        _ifaces = _stack_interface_depths_mm(stack)
        z_plot_min = float(z_mm[0]) * 1e3
        for z_iface_mm, ilabel in _ifaces:
            z_iface_um = z_iface_mm * 1e3
            if z_iface_um <= z_plot_min:
                continue  # skip the surface line (left edge)
            ax_s.axvline(z_iface_um, color="#888888", lw=0.8,
                         linestyle="--", alpha=0.6, zorder=2)
            ax_s.text(
                z_iface_um, 1.0, ilabel,
                transform=ax_s.get_xaxis_transform(),
                color="#888888", fontsize=6, rotation=90,
                va="top", ha="right", alpha=0.8,
            )

    # ── Score matrix heatmap (right panel) ───────────────────────────────────
    if ax_h is not None:
        n_valid = score_matrix.shape[1]
        n_show  = min(top_n_spots, n_valid)

        # Sort by peak score, take top_n_spots
        peak_per_spot = score_matrix.max(axis=0)
        top_idx = np.argsort(peak_per_spot)[::-1][:n_show]
        sub = score_matrix[:, top_idx]

        # Normalise each column independently (0→1) so dim spots are visible
        col_max = sub.max(axis=0, keepdims=True)
        col_max[col_max == 0] = 1.0
        sub_norm = sub / col_max

        im = ax_h.imshow(
            sub_norm.T,
            aspect="auto",
            origin="lower",
            extent=[z_mm[0] * 1e3, z_mm[-1] * 1e3, 0, n_show],
            cmap=cmap_matrix,
            vmin=0, vmax=1,
            interpolation="nearest",
        )
        ax_h.axvline(z_best * 1e3, color="orange", lw=1, linestyle=":",
                     alpha=0.8)
        if stack is not None:
            for z_iface_mm, _ in _ifaces:
                z_iface_um = z_iface_mm * 1e3
                if z_iface_um <= z_plot_min:
                    continue
                ax_h.axvline(z_iface_um, color="#888888", lw=0.8,
                             linestyle="--", alpha=0.6, zorder=2)
        ax_h.set_xlabel("depth  (µm)", color=FG, fontsize=8)
        ax_h.set_ylabel(f"spot rank (top {n_show})", color=FG, fontsize=8)
        ax_h.set_title("Per-spot depth profile (col-normalised)", color=FG, fontsize=9)

        cbar = fig.colorbar(im, ax=ax_h, fraction=0.03, pad=0.01)
        cbar.ax.tick_params(colors=FG, labelsize=6)
        cbar.set_label("score / max", color=FG, fontsize=7)

    fig.tight_layout()
    if out_path is not None:
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=BG)

    return fig, axes_out
