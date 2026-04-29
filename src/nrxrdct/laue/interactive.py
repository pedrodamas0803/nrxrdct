"""
Interactive Laue orientation tool
===================================
Provides a matplotlib-based GUI for manually setting the starting
orientation U0 by rotating simulated spots to match the observed
(segmented) spots.

Typical workflow
----------------
1. Get a rough starting guess from EBSD, LaueTools, or even just zeros::

       U0 = euler_to_U(0, 0, 0, sample_tilt_deg=40)

2. Load the observed spot table from segmentation::

       obs_xy = convert_spotsfile2peaklist("frame_00042.h5")[:, :2]

3. Open the interactive window::

       state = interactive_orientation(crystal, camera, obs_xy, U0)
       # drag sliders until simulated (◆) spots overlap observed (○) spots
       # click "Set as U₀" to bake large adjustments and reset sliders
       # click "✓ Accept" to print and store the final orientation

4. Pass the result to the automatic fitter::

       result = fit_orientation(crystal, camera, obs_xy, state.U)

Backend note
------------
Requires an interactive matplotlib backend.  In a Jupyter notebook run::

    %matplotlib widget     # (ipympl) — recommended
    # or
    %matplotlib qt         # Qt backend for a separate window

In a plain Python script any GUI backend works (Qt, Tk, …).
"""

from __future__ import annotations

import threading
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button, TextBox
from scipy.spatial.transform import Rotation

from .simulation import (
    BM32_KB,
    E_MAX_eV,
    E_MIN_eV,
    F2_THRESHOLD,
    simulate_laue,
    simulate_laue_stack,
)
from .fitting import _extract_sim_xy, _match_spots

# ── colour palette ────────────────────────────────────────────────────────────
_BG      = "#080c14"
_BG2     = "#0d1220"
_FG      = "#ccccee"
_GRAY    = "#4a5070"
_OBS     = "#ffffff"
_SIM     = "#ff6b35"
_MATCH   = "#44dd66"
_MISS    = "#dd4444"
_ACCENT  = "#4fc3f7"
_CRYSTAL = "#ffb347"   # crystal-axis sliders


# ─────────────────────────────────────────────────────────────────────────────
# State container
# ─────────────────────────────────────────────────────────────────────────────


class OrientationState:
    """
    Live orientation state returned by :func:`interactive_orientation`.

    Attributes
    ----------
    U        : (3, 3) ndarray  — current orientation (updates as sliders move).
    U0       : (3, 3) ndarray  — current base orientation (updated by
                                 "Set as U₀" button).
    accepted : bool            — True after the "✓ Accept" button is clicked.
    U_layers : list or None    — per-layer U matrices when a
                                 :class:`~nrxrdct.laue.layers.LayeredCrystal`
                                 was passed.
    """

    def __init__(self, U0: np.ndarray, U0_layers: list | None = None):
        self.U        = U0.copy()
        self.U0       = U0.copy()
        self._U0_orig = U0.copy()
        self.accepted = False
        self.U_layers = None
        self._U0_layers_orig = (
            [u.copy() for u in U0_layers] if U0_layers else None
        )

    def __repr__(self) -> str:
        euler = Rotation.from_matrix(self.U).as_euler("ZXZ", degrees=True)
        return (
            f"OrientationState(accepted={self.accepted},\n"
            f"  Euler(ZXZ) = [{euler[0]:.4f}°, {euler[1]:.4f}°,"
            f" {euler[2]:.4f}°],\n"
            f"  U =\n{np.array2string(self.U, precision=6)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────


def interactive_orientation(
    crystal,
    camera,
    obs_xy: np.ndarray,
    U0: np.ndarray | None = None,
    image: np.ndarray | None = None,
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
    source: str = "bending_magnet",
    source_kwargs: dict | None = None,
    hmax: int = 6,
    f2_thresh: float = F2_THRESHOLD,
    kb_params=BM32_KB,
    structure_model: str = "average",
    max_match_px: float = 30.0,
    top_n_obs: int | None = None,
    top_n_sim: int = 80,
    rot_range_deg: float = 20.0,
    c_rot_range_deg: float = 180.0,
    figsize: tuple = (14, 9),
) -> OrientationState:
    """
    Open an interactive window to manually align the crystal orientation.

    Three sliders rotate the crystal around its own [100], [010], and [001]
    axes (crystal frame).  This is the natural parameterisation for non-cubic
    crystals: tilts around the a/b axes change the sample normal direction
    while the [001] slider sweeps the full in-plane azimuth.

    Parameters
    ----------
    crystal   : xrayutilities Crystal **or** LayeredCrystal
    camera    : Camera
    obs_xy    : (N, 2) ndarray  — observed pixel positions from segmentation.
    U0        : (3, 3) ndarray or None  — starting orientation.
    image     : (Nv, Nh) ndarray or None  — optional background image.
    hmax      : int  — max Miller index (lower = faster, 6 ≈ 20 ms/update).
    max_match_px : float  — pixel radius for match lines.
    top_n_sim : int  — max simulated spots displayed.
    rot_range_deg : float
        Half-range of the [100] and [010] crystal-axis sliders (degrees).
    c_rot_range_deg : float
        Half-range of the [001] crystal-axis slider (degrees).  Defaults to
        180° so the full azimuthal range is accessible in one drag — important
        for non-cubic crystals where the correct in-plane angle can be anywhere
        in 0–360°.

    Returns
    -------
    OrientationState
        ``state.U``  — final orientation (pass to :func:`fit_orientation`).
        ``state.accepted`` — True if "✓ Accept" was clicked.

    Controls
    --------
    Cry [100] / [010]
        Tilt the crystal around its a/b axes.
    Cry [001]
        Rotate in-plane around the c-axis.
    Set as U₀
        Bake current slider values into U₀ and reset to zero — use this
        iteratively for large rotations.
    Reset
        Restore the original U₀ and zero all sliders.
    Center at hkl
        Type Miller indices h k l and click to rotate the crystal so that
        the hkl reflection points toward the detector centre.
    ✓ Accept
        Print the final U matrix and Euler angles; set ``state.accepted=True``.
    """
    from .layers import LayeredCrystal as _LC

    # ── Stack detection ───────────────────────────────────────────────────────
    _is_stack = isinstance(crystal, _LC)

    if _is_stack:
        stack = crystal
        _U0_layers_orig = [lay.U.copy() for lay in stack.all_layers]
        if U0 is None:
            U0 = stack.all_layers[0].U.copy()
    else:
        _U0_layers_orig = None
        if U0 is None:
            raise ValueError("U0 is required for a single-crystal simulation.")

    U0 = np.asarray(U0, dtype=float)
    obs_use = np.asarray(obs_xy, dtype=float)
    if top_n_obs is not None:
        obs_use = obs_use[:top_n_obs]

    state = OrientationState(U0, _U0_layers_orig)
    source_kwargs = source_kwargs or {}

    # ── Simulation helper ─────────────────────────────────────────────────────
    def _simulate(U: np.ndarray) -> list:
        if _is_stack:
            R = U @ np.linalg.inv(state._U0_orig)
            for lay, U0l in zip(stack.all_layers, _U0_layers_orig):
                lay.U = R @ U0l
            spots = simulate_laue_stack(
                stack, camera,
                E_min_eV=E_min_eV, E_max_eV=E_max_eV,
                source=source, source_kwargs=source_kwargs,
                hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
                structure_model=structure_model,
                verbose=False,
            )
            for lay, U0l in zip(stack.all_layers, _U0_layers_orig):
                lay.U = U0l.copy()
            return spots
        else:
            return simulate_laue(
                crystal, U, camera,
                E_min=E_min_eV, E_max=E_max_eV,
                source=source, source_kwargs=source_kwargs,
                hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
            )

    # ── Figure layout ─────────────────────────────────────────────────────────
    # 6 rows × 2 cols
    #   row 0   : detector plot (left) + info text (right)
    #   rows 1-3: crystal-axis sliders ([100], [010], [001])
    #   row 4   : buttons (Reset / Set as U₀ / Accept)
    #   row 5   : "Center at hkl" text-boxes + button
    fig = plt.figure(figsize=figsize, facecolor=_BG)
    fig.canvas.manager.set_window_title("Laue — interactive orientation")

    gs = gridspec.GridSpec(
        6, 2,
        figure=fig,
        left=0.05, right=0.98, bottom=0.03, top=0.96,
        hspace=0.12, wspace=0.06,
        height_ratios=[1.0, 0.055, 0.055, 0.055, 0.075, 0.075],
        width_ratios=[2.6, 1.0],
    )

    ax_det  = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_sa   = fig.add_subplot(gs[1, 0])
    ax_sb   = fig.add_subplot(gs[2, 0])
    ax_sc   = fig.add_subplot(gs[3, 0])
    ax_ph   = fig.add_subplot(gs[4, 0])
    ax_ph2  = fig.add_subplot(gs[5, 0])
    gs_info_bottom = fig.add_subplot(gs[1:, 1])
    gs_info_bottom.set_visible(False)

    # ── Detector axes ─────────────────────────────────────────────────────────
    for ax in (ax_det, ax_info):
        ax.set_facecolor(_BG2)
        ax.tick_params(colors=_GRAY, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(_GRAY)

    ax_det.set_xlim(0, camera.Nh)
    ax_det.set_ylim(camera.Nv, 0)
    ax_det.set_aspect("equal")
    ax_det.set_xlabel("xcam  (px)", color=_FG, fontsize=8)
    ax_det.set_ylabel("ycam  (px)", color=_FG, fontsize=8)
    ax_det.set_title(
        "Laue — interactive orientation   "
        "○ observed   ◆ simulated   — matched   · unmatched",
        color=_FG, fontsize=9, pad=6,
    )

    if image is not None:
        img = np.asarray(image, dtype=float)
        vmax = np.percentile(img[img > 0], 99) if img.max() > 0 else 1.0
        ax_det.imshow(
            np.log1p(img / vmax * 1000),
            origin="upper",
            extent=[0, camera.Nh, camera.Nv, 0],
            cmap="inferno",
            aspect="auto",
            alpha=0.55,
            zorder=0,
        )
    else:
        ax_det.add_patch(plt.Rectangle(
            (0, 0), camera.Nh, camera.Nv,
            linewidth=0.8, edgecolor=_GRAY, facecolor="none", zorder=0,
        ))

    sc_obs = ax_det.scatter(
        obs_use[:, 0], obs_use[:, 1],
        s=45, c="none", edgecolors=_OBS, linewidths=0.9,
        zorder=4, label=f"observed ({len(obs_use)})",
    )
    sc_sim = ax_det.scatter(
        [], [], s=28, c=_SIM, marker="D", linewidths=0,
        zorder=5, label="simulated",
    )
    _lines: list = []
    ax_det.legend(loc="upper right", fontsize=7,
                  facecolor=_BG2, edgecolor=_GRAY, labelcolor=_FG)

    # ── Info panel ────────────────────────────────────────────────────────────
    ax_info.set_axis_off()
    _info_txt = ax_info.text(
        0.06, 0.98, "",
        transform=ax_info.transAxes,
        color=_FG, fontsize=8.5, va="top", family="monospace",
        linespacing=1.55,
    )

    # ── Crystal-axis sliders ──────────────────────────────────────────────────
    _cry_kw = dict(valinit=0.0, color=_CRYSTAL, track_color=_BG)

    s_ca = Slider(ax_sa, "Cry [100]  (a)",
                  valmin=-rot_range_deg, valmax=+rot_range_deg, **_cry_kw)
    s_cb = Slider(ax_sb, "Cry [010]  (b)",
                  valmin=-rot_range_deg, valmax=+rot_range_deg, **_cry_kw)
    s_cc = Slider(ax_sc, "Cry [001]  (c)",
                  valmin=-c_rot_range_deg, valmax=+c_rot_range_deg, **_cry_kw)

    for s in (s_ca, s_cb, s_cc):
        s.label.set_color(_FG)
        s.label.set_fontsize(9)
        s.valtext.set_color(_CRYSTAL)
        s.valtext.set_fontsize(8)
        s.ax.set_facecolor(_BG2)
        for sp in s.ax.spines.values():
            sp.set_edgecolor(_GRAY)

    # ── Buttons ───────────────────────────────────────────────────────────────
    bb  = ax_ph.get_position()
    ax_ph.remove()
    bw  = (bb.width - 0.015) / 3
    bh  = bb.height

    ax_b_reset  = fig.add_axes([bb.x0,                  bb.y0, bw, bh])
    ax_b_setu0  = fig.add_axes([bb.x0 + bw + 0.005,     bb.y0, bw, bh])
    ax_b_accept = fig.add_axes([bb.x0 + 2*bw + 0.010,   bb.y0, bw, bh])

    btn_reset  = Button(ax_b_reset,  "Reset",     color=_BG2,     hovercolor="#1a1f35")
    btn_setu0  = Button(ax_b_setu0,  "Set as U₀", color="#1a2535", hovercolor="#243045")
    btn_accept = Button(ax_b_accept, "✓  Accept",  color="#0d2515", hovercolor="#174025")

    for btn, clr in ((btn_reset, _FG), (btn_setu0, _ACCENT), (btn_accept, _MATCH)):
        btn.label.set_color(clr)
        btn.label.set_fontsize(9)

    # ── "Center at hkl" row ───────────────────────────────────────────────────
    bb2 = ax_ph2.get_position()
    ax_ph2.remove()
    tw   = bb2.width * 0.10
    gap  = 0.006
    bw_c = bb2.width - 3*tw - 4*gap

    ax_tb_h  = fig.add_axes([bb2.x0,                  bb2.y0, tw,   bb2.height])
    ax_tb_k  = fig.add_axes([bb2.x0 +   tw + gap,     bb2.y0, tw,   bb2.height])
    ax_tb_l  = fig.add_axes([bb2.x0 + 2*tw + 2*gap,   bb2.y0, tw,   bb2.height])
    ax_b_ctr = fig.add_axes([bb2.x0 + 3*tw + 3*gap,   bb2.y0, bw_c, bb2.height])

    tb_h = TextBox(ax_tb_h, "h", initial="0", color=_BG2, hovercolor="#1a1f35")
    tb_k = TextBox(ax_tb_k, "k", initial="0", color=_BG2, hovercolor="#1a1f35")
    tb_l = TextBox(ax_tb_l, "l", initial="0", color=_BG2, hovercolor="#1a1f35")
    btn_center = Button(ax_b_ctr, "Center at hkl", color="#1a2535", hovercolor="#243045")

    for tb in (tb_h, tb_k, tb_l):
        tb.label.set_color(_FG)
        tb.label.set_fontsize(9)
        tb.text_disp.set_color(_ACCENT)
        tb.text_disp.set_fontsize(9)
    btn_center.label.set_color(_ACCENT)
    btn_center.label.set_fontsize(9)

    # ── Update ────────────────────────────────────────────────────────────────
    def _do_update() -> None:
        # Compose crystal-axis rotations sequentially.
        # Each axis is expressed in the crystal frame and mapped to the lab
        # frame via the current U — identical to rotate_U_about_crystal_axis.
        U = state.U0
        for angle_deg, cry_ax in (
            (s_ca.val, np.array([1., 0., 0.])),
            (s_cb.val, np.array([0., 1., 0.])),
            (s_cc.val, np.array([0., 0., 1.])),
        ):
            if angle_deg == 0.0:
                continue
            ax_lab = U @ cry_ax
            ax_lab /= np.linalg.norm(ax_lab)
            U = Rotation.from_rotvec(
                np.radians(angle_deg) * ax_lab
            ).as_matrix() @ U

        state.U = U

        dR = U @ np.linalg.inv(state.U0)
        dw = float(np.degrees(Rotation.from_matrix(dR).magnitude()))

        spots  = _simulate(U)
        sim_xy = _extract_sim_xy(spots)[:top_n_sim]

        sc_sim.set_offsets(sim_xy if len(sim_xy) else np.empty((0, 2)))

        for ln in _lines:
            ln.remove()
        _lines.clear()

        n_matched = 0
        rms_px    = float("nan")

        if len(sim_xy) > 0 and len(obs_use) > 0:
            row_ind, col_ind, dist_px = _match_spots(obs_use, sim_xy, max_match_px)
            ok_mask   = dist_px < max_match_px
            n_matched = int(ok_mask.sum())
            if n_matched > 0:
                rms_px = float(np.sqrt((dist_px[ok_mask] ** 2).mean()))
            for r, c, d, ok in zip(row_ind, col_ind, dist_px, ok_mask):
                _lines.append(ax_det.plot(
                    [obs_use[r, 0], sim_xy[c, 0]],
                    [obs_use[r, 1], sim_xy[c, 1]],
                    color=_MATCH if ok else _MISS,
                    lw=0.6, alpha=0.55, zorder=3,
                )[0])

        euler = Rotation.from_matrix(U).as_euler("ZXZ", degrees=True)
        rms_s = f"{rms_px:.1f} px" if np.isfinite(rms_px) else "—"
        rate  = n_matched / max(len(obs_use), 1)

        _info_txt.set_text(
            f"Orientation\n"
            f"{'─'*22}\n"
            f"Euler (ZXZ)\n"
            f"  φ₁ = {euler[0]:+9.3f}°\n"
            f"  Φ  = {euler[1]:+9.3f}°\n"
            f"  φ₂ = {euler[2]:+9.3f}°\n"
            f"\n"
            f"│δω│ = {dw:.4f}°\n"
            f"\n"
            f"Match  ({max_match_px:.0f} px window)\n"
            f"{'─'*22}\n"
            f"  matched : {n_matched} / {len(obs_use)}\n"
            f"  rate    : {rate:.0%}\n"
            f"  rms     : {rms_s}\n"
            f"\n"
            f"Simulated : {len(sim_xy)}\n"
            f"Observed  : {len(obs_use)}\n"
        )

        fig.canvas.draw_idle()

    # ── Button callbacks ──────────────────────────────────────────────────────
    _all_sliders = (s_ca, s_cb, s_cc)

    def _cb_reset(event) -> None:
        state.U0 = state._U0_orig.copy()
        for s in _all_sliders:
            s.reset()

    def _cb_setu0(event) -> None:
        state.U0 = state.U.copy()
        for s in _all_sliders:
            s.reset()

    def _cb_accept(event) -> None:
        state.accepted = True
        if _is_stack:
            R = state.U @ np.linalg.inv(state._U0_orig)
            state.U_layers = [R @ U0l for U0l in _U0_layers_orig]
            for lay, U_new in zip(stack.all_layers, state.U_layers):
                lay.U = U_new.copy()

        euler = Rotation.from_matrix(state.U).as_euler("ZXZ", degrees=True)
        print("\n✓ Orientation accepted")
        print(f"  Euler (ZXZ):  φ₁={euler[0]:.4f}°   Φ={euler[1]:.4f}°   φ₂={euler[2]:.4f}°")
        print(f"  U =\n{np.array2string(state.U, precision=8)}")
        if _is_stack:
            print(f"  Stack updated in place ({len(stack.all_layers)} layers).")
        print("\n  Pass to fitter:  fit_orientation(crystal, camera, obs_xy, state.U)")

    # ── "Center at hkl" callback ──────────────────────────────────────────────
    def _cb_center_hkl(_event) -> None:
        # Read tb.text directly — it always reflects the current display text.
        try:
            h = int(float(tb_h.text))
            k = int(float(tb_k.text))
            l = int(float(tb_l.text))
        except ValueError:
            return
        if h == 0 and k == 0 and l == 0:
            return

        _xtal = crystal.all_layers[0].crystal if _is_stack else crystal
        G_cry = _xtal.Q(h, k, l)
        g_norm = np.linalg.norm(G_cry)
        if g_norm < 1e-12:
            return

        G_lab = state.U @ (G_cry / g_norm)
        G_lab /= np.linalg.norm(G_lab)

        # Target direction: diffracted beam must point toward detector centre.
        # camera.IOlab is in LT2 frame (y // beam); convert to LT (x // beam).
        IO_LT2 = camera.IOlab
        IO_LT  = np.array([IO_LT2[1], -IO_LT2[0], IO_LT2[2]])
        d_hat  = IO_LT / np.linalg.norm(IO_LT)
        ki_hat = np.array([1.0, 0.0, 0.0])
        target = d_hat - ki_hat
        t_norm = np.linalg.norm(target)
        if t_norm < 1e-12:
            return
        target_hat = target / t_norm

        # Minimum-arc rotation: G_lab → target_hat
        ax_r  = np.cross(G_lab, target_hat)
        sin_a = np.linalg.norm(ax_r)
        cos_a = float(np.dot(G_lab, target_hat))
        if sin_a < 1e-10:
            if cos_a > 0:
                R_ctr = np.eye(3)
            else:
                perp = np.array([0., 0., 1.] if abs(G_lab[2]) < 0.9
                                else [0., 1., 0.])
                v = np.cross(G_lab, perp)
                R_ctr = Rotation.from_rotvec(np.pi * v / np.linalg.norm(v)).as_matrix()
        else:
            R_ctr = Rotation.from_rotvec(
                np.arctan2(sin_a, cos_a) * (ax_r / sin_a)
            ).as_matrix()

        state.U0 = R_ctr @ state.U
        for s in _all_sliders:
            s.reset()

    _debounce_timer: list[threading.Timer | None] = [None]

    def _on_slider_changed(_val) -> None:
        if _debounce_timer[0] is not None:
            _debounce_timer[0].cancel()
        _debounce_timer[0] = threading.Timer(0.12, _do_update)
        _debounce_timer[0].start()

    for s in _all_sliders:
        s.on_changed(_on_slider_changed)
    btn_reset.on_clicked(_cb_reset)
    btn_setu0.on_clicked(_cb_setu0)
    btn_accept.on_clicked(_cb_accept)
    btn_center.on_clicked(_cb_center_hkl)

    _do_update()
    plt.show()

    return state
