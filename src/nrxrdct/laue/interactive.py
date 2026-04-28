"""
Interactive Laue orientation tool
===================================
Provides a matplotlib-based GUI for manually setting the starting
orientation U0 by rotating simulated spots to match the observed
(segmented) spots.

Typical workflow
----------------
1. Get a rough starting guess from EBSD, LaueTools, or even just zeros:
       U0 = euler_to_U(0, 0, 0, sample_tilt_deg=40)

2. Load the observed spot table from segmentation:
       obs_xy = convert_spotsfile2peaklist("frame_00042.h5")[:, :2]

3. Open the interactive window:
       state = interactive_orientation(crystal, camera, obs_xy, U0)
       # drag sliders until simulated (◆) spots overlap observed (○) spots
       # click "Set as U₀" to bake large adjustments and reset sliders
       # click "✓ Accept" to print and store the final orientation

4. Pass the result to the automatic fitter:
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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from scipy.spatial.transform import Rotation

from .simulation import (
    BM32_KB,
    E_MAX_eV,
    E_MIN_eV,
    F2_THRESHOLD,
    HMAX,
    simulate_laue,
    simulate_laue_stack,
)
from .fitting import _extract_sim_xy, _match_spots

# ── colour palette matching the rest of the module ───────────────────────────
_BG    = "#080c14"
_BG2   = "#0d1220"
_FG    = "#ccccee"
_GRAY  = "#4a5070"
_OBS   = "#ffffff"   # observed spots
_SIM   = "#ff6b35"   # simulated spots
_MATCH = "#44dd66"   # matched pair lines
_MISS  = "#dd4444"   # unmatched pair lines
_ACCENT = "#4fc3f7"


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
        self._U0_orig = U0.copy()    # full-reset target
        self.accepted = False
        self.U_layers = None         # filled for stack mode
        self._U0_layers_orig = (
            [u.copy() for u in U0_layers] if U0_layers else None
        )

    def __repr__(self) -> str:  # noqa: D105
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
    figsize: tuple = (14, 9),
) -> OrientationState:
    """
    Open an interactive window to manually set the starting orientation U0.

    Parameters
    ----------
    crystal   : xrayutilities Crystal **or** LayeredCrystal
        Crystal structure.  For a ``LayeredCrystal`` the sliders apply a
        single global rotation to all layers simultaneously.
    camera    : Camera
        Detector geometry.
    obs_xy    : (N, 2) ndarray
        Observed spot positions ``[xcam, ycam]`` from segmentation, sorted
        by descending intensity.
    U0        : (3, 3) ndarray or None
        Starting orientation.  For a ``LayeredCrystal`` this defaults to
        the orientation of the first layer.
    image     : (Nv, Nh) ndarray or None
        Optional detector image shown as background (log-scaled for
        contrast).
    hmax      : int
        Maximum Miller index during interactive simulation.  Lower values
        are faster; ``hmax=6`` gives ~20 ms per update.
    max_match_px : float
        Pixel radius for the match lines shown in the plot.
    top_n_sim : int
        Maximum number of simulated spots displayed.
    rot_range_deg : float
        Half-range of each rotation slider (degrees).

    Returns
    -------
    OrientationState
        ``state.U``  — final orientation matrix (ready for :func:`fit_orientation`).
        ``state.accepted`` — True if the "✓ Accept" button was clicked.

    Controls
    --------
    Sliders
        Rotate around the three lab-frame axes (x=beam, y=horizontal, z=vertical).
    Set as U₀
        Bake the current slider values into U₀ and reset sliders to zero,
        allowing iterative large-angle adjustments.
    Reset
        Return sliders (and U₀) to the very first orientation.
    ✓ Accept
        Print the final U matrix and Euler angles; set ``state.accepted=True``.
    """
    from .layers import LayeredCrystal as _LC

    # ── Detect stack mode ─────────────────────────────────────────────────────
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
            # Apply global rotation to all layers (in place, then restore).
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
            # Restore original layer orientations so the stack is not
            # permanently modified until the user clicks Accept.
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
    fig = plt.figure(figsize=figsize, facecolor=_BG)
    fig.canvas.manager.set_window_title("Laue — interactive orientation")

    # 5 rows × 2 cols
    #   row 0 : detector plot (left) + info text (right)
    #   rows 1-3 : sliders (left only)
    #   row 4 : buttons (left only)
    gs = gridspec.GridSpec(
        5, 2,
        figure=fig,
        left=0.05, right=0.98, bottom=0.03, top=0.96,
        hspace=0.12, wspace=0.06,
        height_ratios=[1.0, 0.055, 0.055, 0.055, 0.075],
        width_ratios=[2.6, 1.0],
    )

    ax_det  = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])
    ax_sx   = fig.add_subplot(gs[1, 0])
    ax_sy   = fig.add_subplot(gs[2, 0])
    ax_sz   = fig.add_subplot(gs[3, 0])
    # Buttons are placed manually inside the gs[4,0] bounding box.
    ax_ph   = fig.add_subplot(gs[4, 0])   # placeholder — will be removed
    gs_info_bottom = fig.add_subplot(gs[1:, 1])
    gs_info_bottom.set_visible(False)

    # ── Detector axes ─────────────────────────────────────────────────────────
    for ax in (ax_det, ax_info):
        ax.set_facecolor(_BG2)
        ax.tick_params(colors=_GRAY, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(_GRAY)

    ax_det.set_xlim(0, camera.Nh)
    ax_det.set_ylim(camera.Nv, 0)   # ycam=0 at top, ycam=Nv at bottom
    ax_det.set_aspect("equal")
    ax_det.set_xlabel("xcam  (px)", color=_FG, fontsize=8)
    ax_det.set_ylabel("ycam  (px)", color=_FG, fontsize=8)
    ax_det.set_title(
        "Laue — interactive orientation   "
        "○ observed   ◆ simulated   — matched   · unmatched",
        color=_FG, fontsize=9, pad=6,
    )

    # Optional background image
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
        # Draw detector outline
        rect = plt.Rectangle(
            (0, 0), camera.Nh, camera.Nv,
            linewidth=0.8, edgecolor=_GRAY, facecolor="none", zorder=0,
        )
        ax_det.add_patch(rect)

    # Observed spots — never move
    sc_obs = ax_det.scatter(
        obs_use[:, 0], obs_use[:, 1],
        s=45, c="none", edgecolors=_OBS, linewidths=0.9,
        zorder=4, label=f"observed ({len(obs_use)})",
    )

    # Simulated spots — updated every slider change
    sc_sim = ax_det.scatter(
        [], [], s=28, c=_SIM, marker="D", linewidths=0,
        zorder=5, label="simulated",
    )

    # Match / miss lines container
    _lines: list = []

    # Legend
    leg = ax_det.legend(
        loc="upper right", fontsize=7,
        facecolor=_BG2, edgecolor=_GRAY, labelcolor=_FG,
    )

    # ── Info panel ────────────────────────────────────────────────────────────
    ax_info.set_axis_off()
    _info_txt = ax_info.text(
        0.06, 0.98, "",
        transform=ax_info.transAxes,
        color=_FG, fontsize=8.5, va="top", family="monospace",
        linespacing=1.55,
    )

    # ── Sliders ───────────────────────────────────────────────────────────────
    _slider_kw = dict(
        valmin=-rot_range_deg,
        valmax=+rot_range_deg,
        valinit=0.0,
        color=_ACCENT,
        track_color=_BG,
    )

    s_x = Slider(ax_sx, "Rot X  (beam)", **_slider_kw)
    s_y = Slider(ax_sy, "Rot Y  (horiz)", **_slider_kw)
    s_z = Slider(ax_sz, "Rot Z  (vert)", **_slider_kw)

    for s in (s_x, s_y, s_z):
        s.label.set_color(_FG)
        s.label.set_fontsize(9)
        s.valtext.set_color(_ACCENT)
        s.valtext.set_fontsize(8)
        for ax_s in (s.ax,):
            ax_s.set_facecolor(_BG2)
            for sp in ax_s.spines.values():
                sp.set_edgecolor(_GRAY)

    # ── Buttons ───────────────────────────────────────────────────────────────
    bb = ax_ph.get_position()
    ax_ph.remove()

    bw = (bb.width - 0.015) / 3
    bh = bb.height
    by = bb.y0
    bx0 = bb.x0

    ax_b_reset  = fig.add_axes([bx0,          by, bw, bh])
    ax_b_setu0  = fig.add_axes([bx0 + bw + 0.005,   by, bw, bh])
    ax_b_accept = fig.add_axes([bx0 + 2*bw + 0.010, by, bw, bh])

    btn_reset  = Button(ax_b_reset,  "Reset",      color=_BG2,    hovercolor="#1a1f35")
    btn_setu0  = Button(ax_b_setu0,  "Set as U₀",  color="#1a2535", hovercolor="#243045")
    btn_accept = Button(ax_b_accept, "✓  Accept",   color="#0d2515", hovercolor="#174025")

    for btn, clr in ((btn_reset, _FG), (btn_setu0, _ACCENT), (btn_accept, _MATCH)):
        btn.label.set_color(clr)
        btn.label.set_fontsize(9)

    # ── Update ────────────────────────────────────────────────────────────────
    def _do_update() -> None:
        rv  = np.radians([s_x.val, s_y.val, s_z.val])
        R   = Rotation.from_euler("xyz", rv).as_matrix()
        U   = R @ state.U0
        state.U      = U
        state.rotvec = rv

        # Simulate
        spots  = _simulate(U)
        sim_xy = _extract_sim_xy(spots)[:top_n_sim]

        # Update scatter
        sc_sim.set_offsets(sim_xy if len(sim_xy) else np.empty((0, 2)))

        # Remove old match/miss lines
        for ln in _lines:
            ln.remove()
        _lines.clear()

        # Compute matching and draw lines
        n_matched = 0
        rms_px    = float("nan")

        if len(sim_xy) > 0 and len(obs_use) > 0:
            row_ind, col_ind, dist_px = _match_spots(obs_use, sim_xy, max_match_px)
            ok_mask   = dist_px < max_match_px
            n_matched = int(ok_mask.sum())
            if n_matched > 0:
                rms_px = float(np.sqrt((dist_px[ok_mask] ** 2).mean()))

            for r, c, d, ok in zip(row_ind, col_ind, dist_px, ok_mask):
                ln = ax_det.plot(
                    [obs_use[r, 0], sim_xy[c, 0]],
                    [obs_use[r, 1], sim_xy[c, 1]],
                    color=_MATCH if ok else _MISS,
                    lw=0.6, alpha=0.55, zorder=3,
                )[0]
                _lines.append(ln)

        # Info text
        euler = Rotation.from_matrix(U).as_euler("ZXZ", degrees=True)
        dw    = float(np.degrees(np.linalg.norm(rv)))
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

    def _on_slider(val=None) -> None:
        _do_update()

    # ── Button callbacks ──────────────────────────────────────────────────────
    def _cb_reset(event) -> None:
        """Restore the very first U0 and zero all sliders."""
        state.U0 = state._U0_orig.copy()
        for s in (s_x, s_y, s_z):
            s.reset()          # triggers on_changed → _do_update

    def _cb_setu0(event) -> None:
        """Bake current slider angles into U0, then zero the sliders."""
        state.U0 = state.U.copy()
        for s in (s_x, s_y, s_z):
            s.reset()          # triggers on_changed → _do_update

    def _cb_accept(event) -> None:
        state.accepted = True
        if _is_stack:
            R = state.U @ np.linalg.inv(state._U0_orig)
            state.U_layers = [R @ U0l for U0l in _U0_layers_orig]
            # Write refined matrices into the stack permanently.
            for lay, U_new in zip(stack.all_layers, state.U_layers):
                lay.U = U_new.copy()

        euler = Rotation.from_matrix(state.U).as_euler("ZXZ", degrees=True)
        print("\n✓ Orientation accepted")
        print(f"  Euler (ZXZ):  φ₁={euler[0]:.4f}°   Φ={euler[1]:.4f}°   φ₂={euler[2]:.4f}°")
        print(f"  U =\n{np.array2string(state.U, precision=8)}")
        if _is_stack:
            print(f"  Stack updated in place ({len(stack.all_layers)} layers).")
        print(
            "\n  Pass to fitter:  "
            "fit_orientation(crystal, camera, obs_xy, state.U)"
        )

    s_x.on_changed(_on_slider)
    s_y.on_changed(_on_slider)
    s_z.on_changed(_on_slider)
    btn_reset.on_clicked(_cb_reset)
    btn_setu0.on_clicked(_cb_setu0)
    btn_accept.on_clicked(_cb_accept)

    # Initial render
    _do_update()
    plt.show()

    return state
