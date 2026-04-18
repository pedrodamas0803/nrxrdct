import matplotlib.colors as mcolors
import matplotlib.gridspec as mgridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xrayutilities as xu
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from scipy.spatial.transform import Rotation
from scipy.special import kv

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

    Parameters
    ----------
    spots_bcc, spots_b2 : lists of spot dicts from simulate_laue()
    out_path            : output PNG path
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
    spots, stack, camera, out_path="/mnt/user-data/outputs/laue_layer_contributions.png"
):
    """
    Visualise per-layer intensity contributions across the detector image
    and in 2theta/chi space.

    Produces a figure with one detector-image panel per layer, coloured by
    each layer's fractional intensity contribution at that spot position.
    A summary panel shows the dominant-layer map across the full detector.

    Parameters
    ----------
    spots   : list of dicts from ``layer_contributions_spots()``
    stack   : LayeredCrystal
    camera  : Camera
    out_path: str
    """
    import matplotlib.cm as mcm
    import matplotlib.colors as mcolors
    import matplotlib.gridspec as mgridspec
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [layer.label for layer in stack.layers]
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
    fig = plt.figure(figsize=(5 * n_layers + 4, 10))
    fig.patch.set_facecolor(BG)

    gs = mgridspec.GridSpec(
        2,
        n_layers + 1,
        height_ratios=[1, 1],
        hspace=0.35,
        wspace=0.25,
        left=0.04,
        right=0.97,
        top=0.91,
        bottom=0.06,
    )

    Nh, Nv = camera.Nh, camera.Nv

    # ── Row 0: per-layer detector images ────────────────────────────────────
    for li, (label, col) in enumerate(zip(labels, layer_cols)):
        ax = fig.add_subplot(gs[0, li])
        ax.set_facecolor("#04060e")
        ax.set_xlim(0, Nh)
        ax.set_ylim(Nv, 0)
        ax.set_aspect("auto")
        ax.set_title(f"{label}", color=col, fontsize=8, pad=4)
        ax.set_xlabel("col", color="#7788aa", fontsize=6)
        ax.set_ylabel("row", color="#7788aa", fontsize=6)
        ax.tick_params(colors="#7788aa", labelsize=5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a1f2e")

        # Scatter spots coloured by this layer's fractional contribution
        xs = [s["pix"][0] for s in spots if label in s.get("layer_I_frac", {})]
        ys = [s["pix"][1] for s in spots if label in s.get("layer_I_frac", {})]
        cs = [
            max(-1, min(1, s["layer_I_frac"][label]))
            for s in spots
            if label in s.get("layer_I_frac", {})
        ]
        sz = [
            max(2, 40 * s["intensity"] ** 0.4)
            for s in spots
            if label in s.get("layer_I_frac", {})
        ]

        if xs:
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

        # 2theta contour
        CC, RR, TTH = camera.tth_grid(step=max(1, Nh // 15))
        tc, _ = camera.pixel_to_2theta_chi(camera.xcen, camera.ycen)
        lvls = [tc - 20, tc, tc + 20]
        lvls = [l for l in lvls if TTH.min() < l < TTH.max()]
        if lvls:
            ax.contour(
                CC, RR, TTH, levels=lvls, colors="#1a2a3a", linewidths=0.5, alpha=0.6
            )

    # ── Row 0 last panel: dominant-layer map ─────────────────────────────────
    ax_dom = fig.add_subplot(gs[0, n_layers])
    ax_dom.set_facecolor("#04060e")
    ax_dom.set_xlim(0, Nh)
    ax_dom.set_ylim(Nv, 0)
    ax_dom.set_aspect("auto")
    ax_dom.set_title("Dominant layer", color="#ccccee", fontsize=8, pad=4)
    ax_dom.set_xlabel("col", color="#7788aa", fontsize=6)
    ax_dom.tick_params(colors="#7788aa", labelsize=5)
    for sp in ax_dom.spines.values():
        sp.set_edgecolor("#1a1f2e")

    for s in spots:
        if "layer_I_frac" not in s:
            continue
        dom_idx = max(
            range(n_layers), key=lambda i: s["layer_I_frac"].get(labels[i], -np.inf)
        )
        col = layer_cols[dom_idx]
        sz = max(2, 50 * s["intensity"] ** 0.4)
        ax_dom.scatter(
            s["pix"][0],
            s["pix"][1],
            s=sz,
            color=col,
            alpha=0.8,
            edgecolors="none",
            zorder=3,
        )

    # Legend
    from matplotlib.lines import Line2D

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

    # ── Row 1: 2theta/chi scatter per layer ──────────────────────────────────
    tths_all = [s["tth"] for s in spots]
    chis_all = [s["chi"] for s in spots]
    tth_range = (min(tths_all) - 3, max(tths_all) + 3) if tths_all else (60, 130)
    chi_range = (min(chis_all) - 3, max(chis_all) + 3) if chis_all else (-50, 50)

    for li, (label, col) in enumerate(zip(labels, layer_cols)):
        ax = fig.add_subplot(gs[1, li])
        ax.set_facecolor("#04060e")
        ax.set_xlim(*chi_range)
        ax.set_ylim(*tth_range)
        ax.set_xlabel("chi (deg)", color="#7788aa", fontsize=6)
        ax.set_ylabel("2theta (deg)", color="#7788aa", fontsize=6)
        ax.set_title(f"{label}  –  2theta/chi", color=col, fontsize=7, pad=4)
        ax.tick_params(colors="#7788aa", labelsize=5)
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a1f2e")
        ax.grid(True, ls=":", lw=0.3, color="#181e2e")
        ax.axvline(0, color="#222244", lw=0.5)

        xs = [s["chi"] for s in spots if label in s.get("layer_I_frac", {})]
        ys = [s["tth"] for s in spots if label in s.get("layer_I_frac", {})]
        cs = [
            max(-0.2, min(1, s["layer_I_frac"][label]))
            for s in spots
            if label in s.get("layer_I_frac", {})
        ]
        sz = [
            max(3, 40 * s["intensity"] ** 0.4)
            for s in spots
            if label in s.get("layer_I_frac", {})
        ]
        if xs:
            ax.scatter(
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

    # ── Row 1 last panel: intensity bar chart per layer ───────────────────────
    ax_bar = fig.add_subplot(gs[1, n_layers])
    ax_bar.set_facecolor("#04060e")
    for sp in ax_bar.spines.values():
        sp.set_edgecolor("#1a1f2e")
    ax_bar.tick_params(colors="#7788aa", labelsize=7)

    # Average fractional contribution of each layer across all spots
    avg_fracs = {}
    for label in labels:
        fracs = [s["layer_I_frac"].get(label, 0) for s in spots if "layer_I_frac" in s]
        avg_fracs[label] = np.mean(fracs) * 100 if fracs else 0.0

    bars = ax_bar.barh(
        labels, [avg_fracs[l] for l in labels], color=layer_cols, alpha=0.8
    )
    ax_bar.axvline(0, color="#555566", lw=0.7)
    ax_bar.set_xlabel("Mean intensity fraction (%)", color="#7788aa", fontsize=7)
    ax_bar.set_title("Mean layer contribution", color="#ccccee", fontsize=8, pad=4)
    ax_bar.set_facecolor("#04060e")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(
        0.5,
        0.96,
        f"Per-layer intensity decomposition  |  {stack.name}  |  "
        f"{len(spots)} spots  |  Λ={stack.bilayer_thickness:.1f} Å × {stack.n_rep} rep",
        ha="center",
        fontsize=10,
        color="white",
        fontweight="bold",
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Figure -> {out_path}")
    return out_path


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
        eigenvalues of the 2×2 pixel-space covariance ``cov_pix``, coloured
        by ``sigma_strain_pix`` (semi-major axis).

    Panel B — σ_strain vs 2θ
        Scatter plot of the major (solid) and minor (dashed) broadening
        semi-axes versus 2θ for every spot.  The most-broadened spots are
        labelled with their (hkl) index.

    Panel C — Jacobian heat-map  *(only when* ``jacobians`` *is provided)*
        Rows = top_n most-broadened spots; columns = the 6 Voigt strain
        components.  Cell colour = |∂xcam/∂εᵢⱼ| or |∂ycam/∂εᵢⱼ|
        (RMS of both rows of J), so you can read off which strain components
        most affect which spots.

    Parameters
    ----------
    spots_b : list of dict
        Output of :func:`~nrxrdct.laue.simulation.strain_broadening`.
        Must contain ``'cov_pix'``, ``'sigma_strain_pix'``,
        ``'sigma_strain_minor'``, ``'pix'``, ``'tth'``, ``'hkl'``.
    camera : Camera
        Used for detector dimensions in Panel A.
    jacobians : dict {(h,k,l): ndarray (2,6)}, optional
        Output of :func:`~nrxrdct.laue.simulation.strain_spot_jacobian`.
        When supplied, Panel C is drawn; otherwise it is replaced with a
        colour-bar for Panel A.
    out_path : str, optional
        File path to save the figure.  ``None`` → do not save.
    top_n : int, optional
        Number of most-broadened spots to label / show in Panel C.

    Returns
    -------
    fig : matplotlib.figure.Figure
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

    Parameters
    ----------
    spots_a, spots_b : list of dict
        Spot dicts from :func:`~nrxrdct.laue.simulation.simulate_laue` or
        compatible sources.  Each dict must contain:

        * ``'tth'``         – 2θ in degrees
        * ``'chi'``         – χ in degrees
        * ``'pix'``         – ``(col, row)`` pixel coordinate on the detector
        * ``'E'``           – photon energy in eV
        * ``'hkl'``         – Miller indices tuple ``(h, k, l)``
        * ``'intensity'``   – normalised intensity [0, 1]
        * ``'is_superlattice'`` – bool

    space : ``'angles'`` | ``'detector'``
        ``'angles'``   – x-axis = 2θ (degrees), y-axis = χ (degrees).
        ``'detector'`` – x-axis = column pixel,  y-axis = row pixel.

    label_a, label_b : str
        Legend labels for the two spot sets.

    E_MIN_eV, E_MAX_eV : float
        Energy range for the shared colour-map.

    n_label : int
        Number of strongest spots in each set to annotate with (hkl).

    out_path : str or None
        File path for the saved PNG.  ``None`` → do not save.

    Returns
    -------
    fig : matplotlib.figure.Figure
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
# LAYER SCHEME
# ─────────────────────────────────────────────────────────────────────────────

def plot_layer_scheme(
    stack,
    figsize=(10, 7),
    layer_width=2.2,
    max_reps=6,
    min_display_frac=0.01,
    ax=None,
    out_path=None,
):
    """
    Render a schematic cross-section of a LayeredCrystal stack oriented in
    the LaueTools lab frame (x = beam direction, z = vertical up).

    The view is the XZ side-plane.  Each layer is drawn as a scaled
    parallelogram whose normal is ``stack.n_hat`` projected onto XZ.
    Layers too thin to label inside (< ``min_display_frac`` of total height)
    are annotated with an external callout.

    Parameters
    ----------
    stack : LayeredCrystal
    figsize : (float, float)
    layer_width : float
        Half-width of the layer slabs in display units.
    max_reps : int
        Maximum number of bilayer repetitions to draw.  Stacks with more
        repetitions show an ellipsis annotation.
    min_display_frac : float
        Layers thinner than this fraction of the drawn stack height have
        their label placed outside with a leader line instead of inside.
    ax : matplotlib Axes, optional
        Draw into an existing Axes.  If None a new figure is created.
    out_path : str, optional
        Save figure to this path if provided.

    Returns
    -------
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

    # ── Build layer list (with repetitions) ──────────────────────────────────
    stack._update_offsets()
    n_reps_draw = min(stack.n_rep, max_reps)

    mqw_drawn_ang = n_reps_draw * stack._bilayer_thickness
    has_mqw = bool(stack.layers) and mqw_drawn_ang > 1e-9
    has_buf = bool(stack.buffer_layers)

    if not has_mqw and not has_buf:
        if standalone:
            return fig, ax
        return ax

    # ── Scaling ───────────────────────────────────────────────────────────────
    DISP_H = 5.0   # display height reserved for the MQW region (display units)
    W      = layer_width

    if has_mqw:
        scale = DISP_H / mqw_drawn_ang          # Å → display units (MQW only)
        # Buffer layers are drawn at a fixed height = thickest MQW layer
        max_mqw_ang = max(lyr.thickness for lyr in stack.layers)
        buf_disp_h  = max_mqw_ang * scale
    else:
        # No MQW — fall back to normal scaling for buffer-only stacks
        scale      = DISP_H / stack._buffer_thickness
        buf_disp_h = None   # use real thickness

    # ── Build (layer, s0_disp, s1_disp, is_buffer) list ──────────────────────
    # Display positions are computed independently for buffers (fixed height)
    # and MQW layers (real scale).
    layers_to_draw = []   # (layer, s0, s1, is_buffer)

    # Buffer layers stacked from z_disp=0, each at a fixed display height
    z_disp = 0.0
    for layer in stack.buffer_layers:
        h = buf_disp_h if buf_disp_h is not None else layer.thickness * scale
        layers_to_draw.append((layer, z_disp, z_disp + h, True))
        z_disp += h

    s_mqw_start_disp = z_disp   # where the MQW region begins in display space

    # MQW repeating layers at real scale above the buffer region
    for rep in range(n_reps_draw):
        for layer, z0_local in zip(stack.layers, stack._z_offsets):
            s0 = s_mqw_start_disp + (rep * stack._bilayer_thickness + z0_local) * scale
            s1 = s0 + layer.thickness * scale
            layers_to_draw.append((layer, s0, s1, False))

    total_disp_h = s_mqw_start_disp + (mqw_drawn_ang * scale if has_mqw else 0.0)

    # ── Colour map (unique layer labels) ─────────────────────────────────────
    unique_labels = list(dict.fromkeys(t[0].label for t in layers_to_draw))
    palette = plt.cm.Set2(np.linspace(0.0, 0.85, max(len(unique_labels), 1)))
    cmap = {lbl: palette[i] for i, lbl in enumerate(unique_labels)}

    # ── Draw MQW boundary marker ──────────────────────────────────────────────
    if has_buf and has_mqw:
        pt_l = s_mqw_start_disp * nh_2d - W * th_2d
        pt_r = s_mqw_start_disp * nh_2d + W * th_2d
        ax.plot(
            [pt_l[0], pt_r[0]], [pt_l[1], pt_r[1]],
            color="#ffdd55", linewidth=1.8, linestyle="--", zorder=4,
        )

    # ── Draw layers ───────────────────────────────────────────────────────────
    callout_labels = []   # [(center_xy, text, color)] for thin-layer callouts

    for layer, s0, s1, is_buffer in layers_to_draw:
        ds = s1 - s0
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

        cx = ((s0 + s1) * 0.5) * nh_2d   # centre of parallelogram
        thick_nm = layer.thickness / 10.0

        # Buffer layers show actual thickness and flag as not to scale
        if is_buffer:
            thick_um = layer.thickness / 1e4
            if thick_um >= 0.1:
                thick_str = f"{thick_um:.2f} µm"
            else:
                thick_str = f"{thick_nm:.1f} nm"
            label_str = f"{layer.label}\n{thick_str}\n(not to scale)"
        else:
            label_str = f"{layer.label}\n{thick_nm:.1f} nm"

        if ds >= min_display_frac * DISP_H:
            ax.text(
                cx[0], cx[1], label_str,
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                rotation=0, rotation_mode="anchor",
                color="black", zorder=3, clip_on=True,
            )
        else:
            # Too thin to label inside — queue a callout
            callout_labels.append((cx, label_str, color))

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

    # Clipped-reps marker
    if stack.n_rep > max_reps:
        tip_pos = total_disp_h * nh_2d
        ax.text(
            tip_pos[0] + 0.05, tip_pos[1] + 0.12,
            f"⋮  ({stack.n_rep} repetitions total)",
            color=FG, fontsize=8, va="bottom", zorder=4,
        )

    # ── Incident beam arrow ───────────────────────────────────────────────────
    # The beam (+x) hits the last layer (largest z0 = surface-facing end).
    surf_ctr  = total_disp_h * nh_2d
    beam_tip  = surf_ctr - 0.2 * nh_2d              # slightly inset
    beam_tail = beam_tip + np.array([-2.2, 0.0])     # beam comes from -x
    ax.annotate(
        "", xy=beam_tip, xytext=beam_tail,
        arrowprops=dict(arrowstyle="->", color=COL_DB, lw=2.0, mutation_scale=14),
        zorder=5,
    )
    mid_beam = 0.5 * (beam_tip + beam_tail)
    ax.text(
        mid_beam[0], mid_beam[1] + 0.18,
        "incident beam  (+x)",
        color=COL_DB, fontsize=8, ha="center", va="bottom",
    )

    # ── Surface-normal arrow ──────────────────────────────────────────────────
    n_base = surf_ctr
    n_tip  = surf_ctr + 1.2 * nh_2d
    ax.annotate(
        "", xy=n_tip, xytext=n_base,
        arrowprops=dict(arrowstyle="->", color="white", lw=1.8, mutation_scale=12),
        zorder=5,
    )
    ax.text(
        n_tip[0] + 0.1 * nh_2d[0],
        n_tip[1] + 0.12,
        r"$\hat{n}$  (surface normal)",
        color="white", fontsize=9, ha="center", va="bottom",
    )

    # ── Surface / substrate edge labels ──────────────────────────────────────
    surf_edge_r = surf_ctr + (W + 0.12) * th_2d
    subs_edge_r = 0 * nh_2d + (W + 0.12) * th_2d
    ax.text(surf_edge_r[0], surf_edge_r[1], "surface ▶",
            color="#aaaaaa", fontsize=7.5, va="center", ha="left")
    ax.text(subs_edge_r[0], subs_edge_r[1], "substrate ▶",
            color="#aaaaaa", fontsize=7.5, va="center", ha="left")

    # ── Lab frame axes (bottom-left corner) ──────────────────────────────────
    ax_len  = 0.80
    # Find a comfortable lower-left position
    all_pts = np.array([s * nh_2d + sign * W * th_2d
                        for s in [0, total_disp_h]
                        for sign in [-1, 1]])
    x_min = all_pts[:, 0].min() - 2.8
    z_min = all_pts[:, 1].min() - 0.5
    orig  = np.array([x_min, z_min])

    def _axis_arrow(direction, color, label, label_offset):
        end = orig + ax_len * direction
        ax.annotate(
            "", xy=end, xytext=orig,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.6, mutation_scale=10),
            zorder=6,
        )
        lp = end + label_offset
        ax.text(lp[0], lp[1], label, color=color, fontsize=9,
                fontweight="bold", ha="center", va="center")

    _axis_arrow(np.array([1.0, 0.0]), "#4fc3f7", "x\n(beam)",   np.array([0.22, 0.0]))
    _axis_arrow(np.array([0.0, 1.0]), "#ff9f43", "z\n(up)",     np.array([0.0,  0.22]))

    # y-axis: out-of-plane, shown as ⊙
    ax.plot(*orig, "o", color="#88cc88", ms=8, zorder=6)
    ax.plot(*orig, ".", color="#88cc88", ms=3, zorder=7)
    ax.text(orig[0] - 0.22, orig[1], "y\n(out)", color="#88cc88",
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
    total_um    = stack.total_thickness / 1e4
    reps_note   = (f"  [{n_reps_draw}/{stack.n_rep} reps shown]"
                   if stack.n_rep > max_reps else "")
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

#: Marker cycle for phases — each new unique ``phase_label`` gets the next one.
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
    figsize=(9, 7),
    ax=None,
    out_path: str | None = "laue_stack_spots.png",
):
    """
    Visualise the spot table from :func:`~nrxrdct.laue.simulate_laue_stack`.

    Each phase (unique ``phase_label``) gets a distinct **marker shape**.
    Within a phase, the **marker colour** encodes the satellite / fringe order:

    * ``satellite_order = 0``   — Bragg peak:  brightest colour of the phase palette.
    * ``satellite_order = ±m``  — fringe / superlattice satellite:  progressively
      darker / more saturated shades along the same palette (negative and positive
      orders share the same colour sequence, distinguished by the legend).

    Marker *size* scales with normalised intensity.

    Parameters
    ----------
    spots : list[dict]
        Spot list returned by :func:`~nrxrdct.laue.simulate_laue_stack`.
        Required keys: ``'phase_label'``, ``'satellite_order'``, ``'tth'``,
        ``'chi'``, ``'pix'``, ``'intensity'``.
    space : ``'angles'`` | ``'detector'``
        Coordinate space to plot in.

        * ``'angles'``   — x = 2θ (°), y = χ (°).
        * ``'detector'`` — x = column pixel, y = row pixel.
    n_label : int
        Number of the strongest spots to annotate with ``(hkl)`` labels.
    size_scale : float
        Maximum marker area (``s`` kwarg in ``ax.scatter``).
    min_size : float
        Minimum marker area so that weak spots remain visible.
    figsize : (float, float)
        Figure size in inches (ignored if *ax* is supplied).
    ax : matplotlib.axes.Axes, optional
        Draw into an existing Axes; if *None* a new figure is created.
    out_path : str or None
        Save the figure to this path.  ``None`` → do not save.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
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

    # ── Annotate strongest spots ───────────────────────────────────────────────
    valid = [s for s in spots if _xy(s)[0] is not None]
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
        ax.invert_yaxis()

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
    with *spots*.  The tooltip is implemented via ``motion_notify_event``
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

    * Miller indices ``(hkl)``
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

    Parameters
    ----------
    spots : list[dict]
        Spot list from any ``simulate_laue*`` function.  Required keys:
        ``'tth'``, ``'chi'``, ``'hkl'``, ``'E'``, ``'intensity'``.
        Optional: ``'satellite_order'``, ``'is_superlattice'``,
        ``'phase_label'``.
    i_thresh : float
        Minimum intensity threshold as a fraction of the brightest **Bragg
        peak** (``satellite_order == 0``).  Spots with
        ``intensity < i_thresh * I_bragg_max`` are dropped before plotting.
        Default: ``0.01`` (1 % of the strongest Bragg peak).
        Pass ``0.0`` to show all spots.
    color_by : ``'energy'`` | ``'intensity'`` | ``'phase'``
        Quantity mapped to spot colour:

        * ``'energy'``    — photon energy (plasma colormap)
        * ``'intensity'`` — normalised intensity (viridis colormap)
        * ``'phase'``     — phase label (tab10; requires ``'phase_label'`` key)
    size_scale : float
        Maximum marker area (``s`` in ``scatter``).
    min_size : float
        Minimum marker area so that weak spots remain visible.
    figsize : (float, float)
        Figure size in inches.
    out_path : str or None
        If given, save a **static** PNG snapshot on figure close.
        ``None`` (default) → do not save.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes

    Notes
    -----
    The interactive hover is implemented with matplotlib's built-in event
    system (no extra dependencies).  Call ``plt.show()`` after this function
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
    detector active area are set to ``NaN``.

    Parameters
    ----------
    image : array-like, shape (Nv, Nh)
        Detector image in pixel space (e.g. from :meth:`~Camera.render` or
        a real experimental frame loaded as a numpy array).
    camera : Camera
        Detector geometry used for the forward/inverse projections.
    tth_range : (float, float), optional
        2θ range in degrees ``(tth_min, tth_max)``.  Defaults to the range
        covered by the four detector corners.
    chi_range : (float, float), optional
        χ range in degrees ``(chi_min, chi_max)``.  Defaults to the range
        covered by the four detector corners.
    n_tth, n_chi : int
        Number of output pixels along the 2θ and χ axes.
    interp_order : int
        Interpolation order passed to :func:`scipy.ndimage.map_coordinates`
        (0 = nearest, 1 = bilinear (default), 3 = cubic).

    Returns
    -------
    warped : ndarray, shape (n_chi, n_tth)
        Remapped image.  NaN where the output pixel falls outside the detector.
    tth_ax : ndarray, shape (n_tth,)
        2θ values of the output columns (degrees).
    chi_ax : ndarray, shape (n_chi,)
        χ values of the output rows (degrees).
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
    figsize=(10, 7),
    out_path: str | None = None,
):
    """
    Overlay simulated spot positions on a detector image.

    Two display frames are available via the *frame* parameter:

    * ``'tth_chi'`` *(default)* — the detector image is warped from pixel
      coordinates into an evenly-spaced 2θ / χ grid using
      :func:`warp_image_to_tth_chi`, and simulated spots are overlaid at
      their angular positions.
    * ``'detector'`` — the raw pixel image is displayed without any
      remapping.  Simulated spots are projected to detector pixel coordinates
      (using the ``'pix'`` key when present, or back-projected from their
      2θ / χ angles via :meth:`~Camera.kf_to_pixel`).

    Hovering over a spot in either frame shows the same tooltip (hkl, 2θ,
    χ, energy, intensity, reflection type, phase).

    Parameters
    ----------
    image : array-like, shape (Nv, Nh)
        Detector image in pixel space.
    camera : Camera
        Detector geometry.
    spots : list[dict], optional
        Spot list from :func:`~nrxrdct.laue.simulate_laue`,
        :func:`~nrxrdct.laue.simulate_laue_stack`, or
        :func:`~nrxrdct.laue.simulate_mixed_phases`.
        Required keys: ``'tth'``, ``'chi'``.
    frame : ``'tth_chi'`` | ``'detector'``
        Coordinate frame for the display (see above).
    tth_range, chi_range : (float, float), optional
        Angular range to display (*tth_chi* frame only).
        Defaults to full detector coverage.
    n_tth, n_chi : int
        Warp output resolution (*tth_chi* frame only).
    log_scale : bool
        Apply ``log1p`` scaling to the image before display.
    cmap : str
        Matplotlib colormap for the image.
    spot_marker : str
        Marker style for simulated spots (default ``'+'``).
    spot_size : float
        Marker size for simulated spots.
    spot_color : str or None
        Single colour for all spots.  When ``None``, colours are assigned
        per ``color_spots_by``.
    color_spots_by : ``'phase'`` | ``'order'`` | ``'energy'``
        How to colour spots when ``spot_color`` is ``None``.
    i_thresh : float
        Minimum intensity as a fraction of the brightest Bragg peak
        (``satellite_order == 0``).  Spots below the cutoff are not
        overlaid.  Default: ``0.01`` (1 %).  Pass ``0.0`` to show all spots.
    figsize : (float, float)
    out_path : str or None
        Save figure to this path if provided.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes
    display_image : ndarray
        The image array that was actually plotted — the warped 2θ / χ grid
        when ``frame='tth_chi'``, or the (optionally log-scaled) raw pixel
        image when ``frame='detector'``.
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
