"""
Interactive Laue orientation and calibration tools
===================================================
Provides ipywidgets-based GUIs for:

* :func:`interactive_orientation` — manually align the crystal orientation U
  before running the automatic fitter.
* :func:`interactive_calibration` — simultaneously adjust both the crystal
  orientation and the camera geometry (dd, xcen, ycen, xbet, xgam) to produce
  a good initial guess before running :meth:`~nrxrdct.laue.Camera.fit_calibration`.

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
Requires ``%matplotlib widget`` (ipympl) and ``ipywidgets`` in Jupyter::

    %matplotlib widget

All interactive controls (sliders, buttons) are rendered as native
ipywidgets so they work reliably on remote Jupyter-Slurm servers where
matplotlib canvas click events can be silently dropped.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

# ── colour palette (figure only) ─────────────────────────────────────────────
_BG    = "#080c14"
_BG2   = "#0d1220"
_FG    = "#ccccee"
_GRAY  = "#4a5070"
_OBS   = "#ffffff"
_SIM   = "#ff6b35"
_MATCH = "#44dd66"


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
    figsize: tuple = (14, 6),
) -> OrientationState:
    """
    Open an interactive widget to manually align the crystal orientation.

    The matplotlib figure shows the detector image; all controls (sliders,
    buttons) are rendered as ipywidgets below the figure so they work on
    remote Jupyter-Slurm servers.

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
        180° so the full azimuthal range is accessible in one drag.

    Returns
    -------
    OrientationState
        ``state.U``  — final orientation (pass to :func:`fit_orientation`).
        ``state.accepted`` — True if "✓ Accept" was clicked.
    """
    import ipywidgets as ipw
    from IPython.display import display as _ipy_display

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

    # ── Figure: detector + info panel only (no widget rows) ──────────────────
    with plt.ioff():
        fig = plt.figure(figsize=figsize, facecolor=_BG)
    try:
        fig.canvas.manager.set_window_title("Laue — interactive orientation")
    except Exception:
        pass

    gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        left=0.05, right=0.98, bottom=0.06, top=0.96,
        wspace=0.06,
        width_ratios=[2.6, 1.0],
    )

    ax_det  = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

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

    ax_info.set_axis_off()
    _info_txt = ax_info.text(
        0.06, 0.98, "",
        transform=ax_info.transAxes,
        color=_FG, fontsize=8.5, va="top", family="monospace",
        linespacing=1.55,
    )

    # ── ipywidgets sliders ────────────────────────────────────────────────────
    _sk = dict(
        step=0.02,
        readout_format=".2f",
        continuous_update=False,
        style={"description_width": "110px"},
        layout=ipw.Layout(width="98%"),
    )
    s_ca = ipw.FloatSlider(value=0.0, min=-rot_range_deg,   max=+rot_range_deg,
                           description="Cry [100]  (a)", **_sk)
    s_cb = ipw.FloatSlider(value=0.0, min=-rot_range_deg,   max=+rot_range_deg,
                           description="Cry [010]  (b)", **_sk)
    s_cc = ipw.FloatSlider(value=0.0, min=-c_rot_range_deg, max=+c_rot_range_deg,
                           description="Cry [001]  (c)", **_sk)

    # ── Update ────────────────────────────────────────────────────────────────
    # _updating flag prevents re-entrant calls when we programmatically reset
    # slider values (each .value assignment would otherwise fire _on_slider).
    _updating = [False]

    def _do_update() -> None:
        U = state.U0
        for angle_deg, cry_ax in (
            (s_ca.value, np.array([1., 0., 0.])),
            (s_cb.value, np.array([0., 1., 0.])),
            (s_cc.value, np.array([0., 0., 1.])),
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

        # Reset all observed markers to white
        sc_obs.set_edgecolors([_OBS] * len(obs_use))

        if len(sim_xy) > 0 and len(obs_use) > 0:
            row_ind, col_ind, dist_px = _match_spots(obs_use, sim_xy, max_match_px)
            ok_mask   = dist_px < max_match_px
            n_matched = int(ok_mask.sum())
            if n_matched > 0:
                rms_px = float(np.sqrt((dist_px[ok_mask] ** 2).mean()))

            # Green edge on matched observed spots; green line to simulated partner.
            # Unmatched pairs get no line — long red lines across the detector are
            # misleading when obs > sim (unmatched obs have no simulated counterpart).
            edge_colors = [_OBS] * len(obs_use)
            for r, c, ok in zip(row_ind, col_ind, ok_mask):
                if ok:
                    edge_colors[r] = _MATCH
                    _lines.append(ax_det.plot(
                        [obs_use[r, 0], sim_xy[c, 0]],
                        [obs_use[r, 1], sim_xy[c, 1]],
                        color=_MATCH, lw=0.7, alpha=0.6, zorder=3,
                    )[0])
            sc_obs.set_edgecolors(edge_colors)

        euler = Rotation.from_matrix(U).as_euler("ZXZ", degrees=True)
        rms_s = f"{rms_px:.1f} px" if np.isfinite(rms_px) else "—"
        # Rate relative to the smaller of obs/sim — fairer when top_n_sim < n_obs
        rate  = n_matched / max(min(len(obs_use), len(sim_xy)), 1)

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

    def _reset_sliders() -> None:
        _updating[0] = True
        s_ca.value = 0.0
        s_cb.value = 0.0
        s_cc.value = 0.0
        _updating[0] = False
        _do_update()

    def _on_slider(change) -> None:
        if not _updating[0]:
            _do_update()

    for s in (s_ca, s_cb, s_cc):
        s.observe(_on_slider, names="value")

    # ── ipywidgets buttons ────────────────────────────────────────────────────
    _bkw = dict(layout=ipw.Layout(width="130px", height="32px"))
    btn_reset  = ipw.Button(description="Reset",      button_style="warning", **_bkw)
    btn_setu0  = ipw.Button(description="Set as U₀",  button_style="info",    **_bkw)
    btn_accept = ipw.Button(description="✓  Accept",  button_style="success", **_bkw)

    def _cb_reset(_b) -> None:
        state.U0 = state._U0_orig.copy()
        _reset_sliders()

    def _cb_setu0(_b) -> None:
        state.U0 = state.U.copy()
        _reset_sliders()

    def _cb_accept(_b) -> None:
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

    btn_reset.on_click(_cb_reset)
    btn_setu0.on_click(_cb_setu0)
    btn_accept.on_click(_cb_accept)

    # ── "Center at hkl" preset buttons ───────────────────────────────────────
    # Planes: cubic axes + diagonal + hexagonal first-order {10-10} and
    # second-order {11-20} prismatic planes (3-index notation).
    _presets: list[tuple[str, tuple[int, int, int]]] = [
        ("100",  ( 1,  0,  0)),
        ("010",  ( 0,  1,  0)),
        ("001",  ( 0,  0,  1)),
        ("111",  ( 1,  1,  1)),
        ("-110", (-1,  1,  0)),   # hex {10-10} first-order prism
        ("110",  ( 1,  1,  0)),   # hex {11-20} second-order prism
    ]

    def _make_center_cb(hkl: tuple[int, int, int]):
        h, k, l = hkl
        def _cb(_b) -> None:
            _xtal = crystal.all_layers[0].crystal if _is_stack else crystal
            G_cry = _xtal.Q(h, k, l)
            g_norm = np.linalg.norm(G_cry)
            if g_norm < 1e-12:
                return

            G_lab = state.U @ (G_cry / g_norm)
            G_lab /= np.linalg.norm(G_lab)

            # Target: diffracted beam toward detector centre.
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
                    R_ctr = Rotation.from_rotvec(
                        np.pi * v / np.linalg.norm(v)
                    ).as_matrix()
            else:
                R_ctr = Rotation.from_rotvec(
                    np.arctan2(sin_a, cos_a) * (ax_r / sin_a)
                ).as_matrix()

            state.U0 = R_ctr @ state.U
            _reset_sliders()
        return _cb

    _ckw = dict(layout=ipw.Layout(width="68px", height="28px"))
    _center_btns = []
    for _lbl, _hkl in _presets:
        _b = ipw.Button(description=_lbl, **_ckw)
        _b.on_click(_make_center_cb(_hkl))
        _center_btns.append(_b)

    # ── Layout and display ────────────────────────────────────────────────────
    _do_update()

    _controls = ipw.VBox([
        s_ca, s_cb, s_cc,
        ipw.HBox(
            [btn_reset, btn_setu0, btn_accept],
            layout=ipw.Layout(margin="8px 0 6px 0", gap="6px"),
        ),
        ipw.HBox(
            [ipw.HTML("<b>Center at hkl:</b>",
                      layout=ipw.Layout(align_self="center", margin="0 10px 0 0"))]
            + _center_btns,
            layout=ipw.Layout(gap="4px"),
        ),
    ], layout=ipw.Layout(width="100%", padding="4px 8px"))

    _ipy_display(ipw.VBox([fig.canvas, _controls]))

    return state


# ─────────────────────────────────────────────────────────────────────────────
# Calibration state container
# ─────────────────────────────────────────────────────────────────────────────


class CalibrationState:
    """
    Live calibration state returned by :func:`interactive_calibration`.

    Attributes
    ----------
    camera   : Camera  — current camera (updates as sliders move).
    camera0  : Camera  — current base camera (updated by "Set as reference").
    U        : (3, 3) ndarray  — current orientation.
    U0       : (3, 3) ndarray  — current base orientation.
    accepted : bool  — True after the "✓ Accept" button is clicked.
    """

    def __init__(self, camera0, U0: np.ndarray):
        self.camera    = camera0
        self.camera0   = camera0
        self._cam_orig = camera0
        self.U         = U0.copy()
        self.U0        = U0.copy()
        self._U0_orig  = U0.copy()
        self.accepted  = False

    def __repr__(self) -> str:
        cam = self.camera
        return (
            f"CalibrationState(accepted={self.accepted},\n"
            f"  Camera(dd={cam.dd:.4g}, xcen={cam.xcen:.4g},"
            f" ycen={cam.ycen:.4g},\n"
            f"         xbet={cam.xbet:.4g}, xgam={cam.xgam:.4g}),\n"
            f"  U =\n{np.array2string(self.U, precision=6)})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Interactive calibration entry point
# ─────────────────────────────────────────────────────────────────────────────


def interactive_calibration(
    crystal,
    camera,
    obs_xy: np.ndarray,
    U0: np.ndarray,
    image: np.ndarray | None = None,
    E_min_eV: float = E_MIN_eV,
    E_max_eV: float = E_MAX_eV,
    source: str = "bending_magnet",
    source_kwargs: dict | None = None,
    hmax: int = 6,
    f2_thresh: float = F2_THRESHOLD,
    kb_params=BM32_KB,
    max_match_px: float = 30.0,
    top_n_sim: int = 80,
    rot_range_deg: float = 20.0,
    c_rot_range_deg: float = 180.0,
    dd_range: float = 10.0,
    cen_range_px: float = 150.0,
    angle_range_deg: float = 2.0,
    space: str = "angular",
    figsize: tuple = (14, 6),
) -> CalibrationState:
    """
    Open an interactive widget to simultaneously adjust crystal orientation
    and camera geometry as a starting point for :meth:`Camera.fit_calibration`.

    Two columns of sliders appear below the detector view:

    * **Orientation** — incremental rotations around the three crystal axes
      (same as :func:`interactive_orientation`).
    * **Camera geometry** — additive deltas for ``dd``, ``xcen``, ``ycen``,
      ``xbet``, ``xgam`` relative to the current reference camera.

    "Set as reference" bakes the current camera + orientation as the new base
    and resets all sliders to zero.  "✓ Accept" prints the final parameters
    and a ready-to-run :meth:`~Camera.fit_calibration` call.

    Parameters
    ----------
    crystal : xrayutilities Crystal
        Calibration standard crystal (not LayeredCrystal).
    camera : Camera
        Initial camera geometry.
    obs_xy : (N, 2) ndarray
        Observed spot pixel positions from segmentation.
    U0 : (3, 3) ndarray
        Initial crystal orientation matrix.
    image : (Nv, Nh) ndarray or None
        Optional background detector image (log-scaled for display).
    hmax : int
        Max Miller index (lower = faster; 6 ≈ 20 ms per update).
    max_match_px : float
        Pixel radius for match lines and match-rate reporting.
    top_n_sim : int
        Maximum number of simulated spots rendered.
    rot_range_deg : float
        Half-range of the [100] / [010] orientation sliders (°).
    c_rot_range_deg : float
        Half-range of the [001] orientation slider (°).
    dd_range : float
        Half-range of the Δ dd slider (mm).
    cen_range_px : float
        Half-range of the Δ xcen / Δ ycen sliders (px).
    angle_range_deg : float
        Half-range of the Δ xbet / Δ xgam sliders (°).
    space : ``'angular'`` or ``'detector'``
        Coordinate frame for the main panel.  ``'angular'`` (default) plots
        2θ (x) vs χ (y) in degrees.  ``'detector'`` plots raw pixel positions.
        Matching is always done in detector (pixel) space regardless of this flag.

    Returns
    -------
    CalibrationState
        ``state.camera``   — pass to :meth:`~Camera.fit_calibration`.
        ``state.U``        — pass to :meth:`~Camera.fit_calibration`.
        ``state.accepted`` — True if "✓ Accept" was clicked.
    """
    import ipywidgets as ipw
    from IPython.display import display as _ipy_display
    from .camera import Camera as _Camera

    U0 = np.asarray(U0, dtype=float)
    obs_use = np.asarray(obs_xy, dtype=float)
    state = CalibrationState(camera, U0)
    source_kwargs = source_kwargs or {}

    # ── Simulation ────────────────────────────────────────────────────────────
    def _simulate(U: np.ndarray, cam) -> list:
        return simulate_laue(
            crystal, U, cam,
            E_min=E_min_eV, E_max=E_max_eV,
            source=source, source_kwargs=source_kwargs,
            hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
        )

    # ── Figure ────────────────────────────────────────────────────────────────
    with plt.ioff():
        fig = plt.figure(figsize=figsize, facecolor=_BG)
    try:
        fig.canvas.manager.set_window_title("Laue — interactive calibration")
    except Exception:
        pass

    gs = gridspec.GridSpec(
        1, 2,
        figure=fig,
        left=0.05, right=0.98, bottom=0.06, top=0.96,
        wspace=0.06,
        width_ratios=[2.6, 1.0],
    )
    ax_det  = fig.add_subplot(gs[0, 0])
    ax_info = fig.add_subplot(gs[0, 1])

    for ax in (ax_det, ax_info):
        ax.set_facecolor(_BG2)
        ax.tick_params(colors=_GRAY, labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor(_GRAY)

    ax_det.set_title(
        "Laue — interactive calibration   "
        "○ observed   ◆ simulated   — matched",
        color=_FG, fontsize=9, pad=6,
    )

    if space == "detector":
        ax_det.set_xlim(0, camera.Nh)
        ax_det.set_ylim(camera.Nv, 0)
        ax_det.set_aspect("equal")
        ax_det.set_xlabel("xcam  (px)", color=_FG, fontsize=8)
        ax_det.set_ylabel("ycam  (px)", color=_FG, fontsize=8)
        if image is not None:
            img_arr = np.asarray(image, dtype=float)
            vmax = np.percentile(img_arr[img_arr > 0], 99) if img_arr.max() > 0 else 1.0
            ax_det.imshow(
                np.log1p(img_arr / vmax * 1000),
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
    else:
        ax_det.set_aspect("auto")
        ax_det.set_xlabel("2θ  (°)", color=_FG, fontsize=8)
        ax_det.set_ylabel("χ  (°)", color=_FG, fontsize=8)
        ax_det.grid(True, ls=":", lw=0.35, color="#181e2e", zorder=0)

    # Initial observed scatter; angular mode will update positions each frame
    # since pixel→angle mapping depends on the current camera.
    _uf0 = camera.pixel_to_kf(obs_use[:, 0], obs_use[:, 1])
    _obs_init = (
        obs_use if space == "detector"
        else np.column_stack([
            np.degrees(np.arccos(np.clip(_uf0[:, 0], -1.0, 1.0))),
            np.degrees(np.arctan2(_uf0[:, 1], _uf0[:, 2] + 1e-17)),
        ])
    )
    sc_obs = ax_det.scatter(
        _obs_init[:, 0], _obs_init[:, 1],
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

    ax_info.set_axis_off()
    _info_txt = ax_info.text(
        0.06, 0.98, "",
        transform=ax_info.transAxes,
        color=_FG, fontsize=8.0, va="top", family="monospace",
        linespacing=1.5,
    )

    # ── Sliders ───────────────────────────────────────────────────────────────
    _sk_o = dict(
        step=0.02, readout_format=".2f", continuous_update=False,
        style={"description_width": "120px"},
        layout=ipw.Layout(width="98%"),
    )
    _sk_c = dict(
        continuous_update=False,
        style={"description_width": "120px"},
        layout=ipw.Layout(width="98%"),
    )

    # Orientation — delta rotations around crystal axes
    s_ca = ipw.FloatSlider(value=0.0, min=-rot_range_deg,    max=+rot_range_deg,
                           description="Cry [100]  (a)", **_sk_o)
    s_cb = ipw.FloatSlider(value=0.0, min=-rot_range_deg,    max=+rot_range_deg,
                           description="Cry [010]  (b)", **_sk_o)
    s_cc = ipw.FloatSlider(value=0.0, min=-c_rot_range_deg,  max=+c_rot_range_deg,
                           description="Cry [001]  (c)", **_sk_o)

    # Camera geometry — additive deltas from camera0
    s_dd   = ipw.FloatSlider(value=0.0, min=-dd_range,        max=+dd_range,
                             step=0.05, readout_format=".2f",
                             description="Δ dd  (mm)", **_sk_c)
    s_xcen = ipw.FloatSlider(value=0.0, min=-cen_range_px,    max=+cen_range_px,
                             step=0.5,  readout_format=".1f",
                             description="Δ xcen  (px)", **_sk_c)
    s_ycen = ipw.FloatSlider(value=0.0, min=-cen_range_px,    max=+cen_range_px,
                             step=0.5,  readout_format=".1f",
                             description="Δ ycen  (px)", **_sk_c)
    s_xbet = ipw.FloatSlider(value=0.0, min=-angle_range_deg, max=+angle_range_deg,
                             step=0.01, readout_format=".3f",
                             description="Δ xbet  (°)", **_sk_c)
    s_xgam = ipw.FloatSlider(value=0.0, min=-angle_range_deg, max=+angle_range_deg,
                             step=0.01, readout_format=".3f",
                             description="Δ xgam  (°)", **_sk_c)

    _all_sliders = (s_ca, s_cb, s_cc, s_dd, s_xcen, s_ycen, s_xbet, s_xgam)

    # ── Update loop ───────────────────────────────────────────────────────────
    _updating = [False]

    def _do_update() -> None:
        # Current orientation: U0 + slider rotations around crystal axes
        U = state.U0
        for angle_deg, cry_ax in (
            (s_ca.value, np.array([1., 0., 0.])),
            (s_cb.value, np.array([0., 1., 0.])),
            (s_cc.value, np.array([0., 0., 1.])),
        ):
            if angle_deg == 0.0:
                continue
            ax_lab = U @ cry_ax
            ax_lab /= np.linalg.norm(ax_lab)
            U = Rotation.from_rotvec(np.radians(angle_deg) * ax_lab).as_matrix() @ U
        state.U = U

        # Current camera: camera0 + delta sliders
        cam0 = state.camera0
        cam = _Camera(
            dd=cam0.dd       + s_dd.value,
            xcen=cam0.xcen   + s_xcen.value,
            ycen=cam0.ycen   + s_ycen.value,
            xbet=cam0.xbet   + s_xbet.value,
            xgam=cam0.xgam   + s_xgam.value,
            pixelsize=cam0.pixel_mm,
            n_pix_h=cam0.Nh,
            n_pix_v=cam0.Nv,
            kf_direction=cam0.kf_direction,
        )
        state.camera = cam

        dR = U @ np.linalg.inv(state.U0)
        dw = float(np.degrees(Rotation.from_matrix(dR).magnitude()))

        spots  = _simulate(U, cam)
        sim_xy = _extract_sim_xy(spots)[:top_n_sim]

        # Build display coordinates (angular or detector)
        if space == "angular":
            uf = cam.pixel_to_kf(obs_use[:, 0], obs_use[:, 1])
            tth_o = np.degrees(np.arccos(np.clip(uf[:, 0], -1.0, 1.0)))
            chi_o = np.degrees(np.arctan2(uf[:, 1], uf[:, 2] + 1e-17))
            obs_disp = np.column_stack([tth_o, chi_o])
            on_det = [s for s in spots if s.get("pix") is not None][:top_n_sim]
            sim_disp = (
                np.array([[s["tth"], s["chi"]] for s in on_det])
                if on_det else np.empty((0, 2))
            )
            sc_obs.set_offsets(obs_disp)
        else:
            obs_disp = obs_use
            sim_disp = sim_xy

        sc_sim.set_offsets(sim_disp if len(sim_disp) else np.empty((0, 2)))

        for ln in _lines:
            ln.remove()
        _lines.clear()

        n_matched = 0
        rms_px    = float("nan")
        sc_obs.set_edgecolors([_OBS] * len(obs_use))

        if len(sim_xy) > 0 and len(obs_use) > 0:
            row_ind, col_ind, dist_px = _match_spots(obs_use, sim_xy, max_match_px)
            ok_mask   = dist_px < max_match_px
            n_matched = int(ok_mask.sum())
            if n_matched > 0:
                rms_px = float(np.sqrt((dist_px[ok_mask] ** 2).mean()))

            edge_colors = [_OBS] * len(obs_use)
            for r, c, ok in zip(row_ind, col_ind, ok_mask):
                if ok:
                    edge_colors[r] = _MATCH
                    _lines.append(ax_det.plot(
                        [obs_disp[r, 0], sim_disp[c, 0]],
                        [obs_disp[r, 1], sim_disp[c, 1]],
                        color=_MATCH, lw=0.7, alpha=0.6, zorder=3,
                    )[0])
            sc_obs.set_edgecolors(edge_colors)

        rms_s = f"{rms_px:.1f} px" if np.isfinite(rms_px) else "—"
        rate  = n_matched / max(min(len(obs_use), len(sim_xy)), 1)
        u_rows = [
            f"  [{U[i,0]:+.4f}  {U[i,1]:+.4f}  {U[i,2]:+.4f}]"
            for i in range(3)
        ]

        _info_txt.set_text(
            f"Camera\n"
            f"{'─' * 22}\n"
            f"  dd   ={cam.dd:9.3f} mm\n"
            f"  xcen ={cam.xcen:9.2f} px\n"
            f"  ycen ={cam.ycen:9.2f} px\n"
            f"  xbet ={cam.xbet:9.4f} °\n"
            f"  xgam ={cam.xgam:9.4f} °\n"
            f"\n"
            f"Orientation  (U matrix)\n"
            f"{'─' * 22}\n"
            + "\n".join(u_rows) + "\n"
            f"  |δω| = {dw:.4f}°\n"
            f"\n"
            f"Match  ({max_match_px:.0f} px window)\n"
            f"{'─' * 22}\n"
            f"  matched : {n_matched} / {len(obs_use)}\n"
            f"  rate    : {rate:.0%}\n"
            f"  rms     : {rms_s}\n"
            f"\n"
            f"  sim={len(sim_xy)}   obs={len(obs_use)}\n"
        )

        fig.canvas.draw_idle()

    def _reset_sliders() -> None:
        _updating[0] = True
        for s in _all_sliders:
            s.value = 0.0
        _updating[0] = False
        _do_update()

    def _on_slider(change) -> None:  # noqa: ARG001
        if not _updating[0]:
            _do_update()

    for s in _all_sliders:
        s.observe(_on_slider, names="value")

    # ── Buttons ───────────────────────────────────────────────────────────────
    _bkw = dict(layout=ipw.Layout(width="160px", height="32px"))
    btn_reset  = ipw.Button(description="Reset",             button_style="warning", **_bkw)
    btn_setref = ipw.Button(description="Set as reference",  button_style="info",    **_bkw)
    btn_accept = ipw.Button(description="✓  Accept",         button_style="success", **_bkw)

    def _cb_reset(_b) -> None:  # noqa: ARG001
        state.U0      = state._U0_orig.copy()
        state.camera0 = state._cam_orig
        _reset_sliders()

    def _cb_setref(_b) -> None:  # noqa: ARG001
        state.U0      = state.U.copy()
        state.camera0 = state.camera
        _reset_sliders()

    def _cb_accept(_b) -> None:  # noqa: ARG001
        state.accepted = True
        cam = state.camera
        print("\n✓ Calibration initial guess accepted")
        print(f"  Camera:")
        print(f"    dd={cam.dd:.5g}  xcen={cam.xcen:.5g}  ycen={cam.ycen:.5g}")
        print(f"    xbet={cam.xbet:.5g}  xgam={cam.xgam:.5g}")
        print(f"  U =\n{np.array2string(state.U, precision=8)}")
        print(
            f"\n  Re-create camera:\n"
            f"    Camera(dd={cam.dd:.5g}, xcen={cam.xcen:.5g},"
            f" ycen={cam.ycen:.5g},\n"
            f"           xbet={cam.xbet:.5g}, xgam={cam.xgam:.5g},"
            f" pixelsize={cam.pixel_mm:.6g},\n"
            f"           n_pix_h={cam.Nh}, n_pix_v={cam.Nv},"
            f" kf_direction={cam.kf_direction!r})"
        )
        print(
            "\n  Pass to fitter:\n"
            "    result = state.camera.fit_calibration(crystal, state.U, obs_xy)"
        )

    btn_reset.on_click(_cb_reset)
    btn_setref.on_click(_cb_setref)
    btn_accept.on_click(_cb_accept)

    # ── Quick-fit button ──────────────────────────────────────────────────────
    btn_fit = ipw.Button(
        description="⚡ Quick Fit", button_style="primary", **_bkw
    )

    def _cb_fit(_b) -> None:  # noqa: ARG001
        import asyncio
        import queue as _qmod
        import threading
        from scipy.optimize import minimize
        from scipy.spatial import cKDTree
        from .simulation import precompute_allowed_hkl

        fit_params = ["dd", "xcen", "ycen", "xbet", "xgam"]
        cam0  = state.camera       # snapshot at click time
        U_fit = state.U.copy()
        x0    = np.array([getattr(cam0, p) for p in fit_params], dtype=float)

        _allowed = precompute_allowed_hkl(crystal, hmax, f2_thresh=f2_thresh)
        _q: _qmod.Queue = _qmod.Queue()

        def _build(x: np.ndarray):
            kw = dict(
                dd=cam0.dd, xcen=cam0.xcen, ycen=cam0.ycen,
                xbet=cam0.xbet, xgam=cam0.xgam,
                pixelsize=cam0.pixel_mm, n_pix_h=cam0.Nh, n_pix_v=cam0.Nv,
                kf_direction=cam0.kf_direction,
            )
            for i, p in enumerate(fit_params):
                kw[p] = float(x[i])
            return _Camera(**kw)

        def _cost(x: np.ndarray) -> float:
            try:
                spots = simulate_laue(
                    crystal, U_fit, _build(x),
                    E_min=E_min_eV, E_max=E_max_eV,
                    source=source, source_kwargs=source_kwargs,
                    hmax=hmax, f2_thresh=f2_thresh, kb_params=kb_params,
                    allowed_hkl=_allowed,
                )
            except Exception:
                return float(max_match_px ** 2)
            sim_c = np.array([s["pix"] for s in spots if s.get("pix") is not None])
            if not len(sim_c):
                return float(max_match_px ** 2)
            dists, _ = cKDTree(sim_c).query(obs_use, k=1)
            return float(np.mean(np.minimum(dists, max_match_px) ** 2))

        _n = [0]

        def _opt_cb(xk: np.ndarray) -> None:
            _n[0] += 1
            if _n[0] % 3 == 0:   # update display every 3 Nelder-Mead iterations
                _q.put(xk.copy())

        def _optimize() -> None:
            _steps = {"dd": 2.0, "xcen": 50.0, "ycen": 50.0, "xbet": 0.5, "xgam": 0.5}
            n = len(x0)
            simplex = np.tile(x0, (n + 1, 1))
            for i, p in enumerate(fit_params):
                simplex[i + 1, i] += _steps[p]
            res = minimize(
                _cost, x0, method="Nelder-Mead",
                callback=_opt_cb,
                options={"maxiter": 500, "xatol": 0.02, "fatol": 1e-4,
                         "initial_simplex": simplex},
            )
            _q.put(res.x.copy())
            _q.put(None)  # sentinel: done

        async def _ui_loop() -> None:
            btn_fit.disabled = True
            btn_fit.description = "Fitting…"
            t = threading.Thread(target=_optimize, daemon=True)
            t.start()
            while True:
                await asyncio.sleep(0.15)  # poll every 150 ms
                latest, done = None, False
                while True:
                    try:
                        item = _q.get_nowait()
                        if item is None:
                            done = True
                            break
                        latest = item
                    except _qmod.Empty:
                        break
                if latest is not None:
                    state.camera0 = _build(latest)
                    state.U0 = U_fit.copy()
                    _updating[0] = True
                    for s in _all_sliders:
                        s.value = 0.0
                    _updating[0] = False
                    _do_update()
                if done:
                    btn_fit.description = "⚡ Quick Fit"
                    btn_fit.disabled = False
                    return

        asyncio.ensure_future(_ui_loop())

    btn_fit.on_click(_cb_fit)

    # ── "Center at hkl" preset buttons ───────────────────────────────────────
    _presets: list[tuple[str, tuple[int, int, int]]] = [
        ("100",  ( 1,  0,  0)),
        ("010",  ( 0,  1,  0)),
        ("001",  ( 0,  0,  1)),
        ("111",  ( 1,  1,  1)),
        ("-110", (-1,  1,  0)),
        ("110",  ( 1,  1,  0)),
    ]

    def _make_center_cb(hkl: tuple[int, int, int]):
        h, k, l_idx = hkl
        def _cb(_b) -> None:  # noqa: ARG001
            G_cry  = crystal.Q(h, k, l_idx)
            g_norm = np.linalg.norm(G_cry)
            if g_norm < 1e-12:
                return

            G_lab = state.U @ (G_cry / g_norm)
            G_lab /= np.linalg.norm(G_lab)

            # Target: momentum-transfer direction pointing at the detector centre.
            # camera.IOlab is in LT2 frame; convert to LT (x // ki).
            IO_LT2     = state.camera.IOlab
            IO_LT      = np.array([IO_LT2[1], -IO_LT2[0], IO_LT2[2]])
            d_hat      = IO_LT / np.linalg.norm(IO_LT)
            target     = d_hat - np.array([1.0, 0.0, 0.0])
            t_norm     = np.linalg.norm(target)
            if t_norm < 1e-12:
                return
            target_hat = target / t_norm

            ax_r  = np.cross(G_lab, target_hat)
            sin_a = np.linalg.norm(ax_r)
            cos_a = float(np.dot(G_lab, target_hat))
            if sin_a < 1e-10:
                if cos_a > 0:
                    R_ctr = np.eye(3)
                else:
                    perp  = np.array([0., 0., 1.] if abs(G_lab[2]) < 0.9
                                     else [0., 1., 0.])
                    v     = np.cross(G_lab, perp)
                    R_ctr = Rotation.from_rotvec(
                        np.pi * v / np.linalg.norm(v)
                    ).as_matrix()
            else:
                R_ctr = Rotation.from_rotvec(
                    np.arctan2(sin_a, cos_a) * (ax_r / sin_a)
                ).as_matrix()

            state.U0 = R_ctr @ state.U
            _reset_sliders()
        return _cb

    _ckw = dict(layout=ipw.Layout(width="68px", height="28px"))
    _center_btns = []
    for _lbl, _hkl in _presets:
        _b = ipw.Button(description=_lbl, **_ckw)
        _b.on_click(_make_center_cb(_hkl))
        _center_btns.append(_b)

    # ── Layout and display ────────────────────────────────────────────────────
    _do_update()

    _col_orient = ipw.VBox(
        [ipw.HTML("<b style='font-size:13px'>Orientation</b>",
                  layout=ipw.Layout(margin="0 0 4px 0")),
         s_ca, s_cb, s_cc],
        layout=ipw.Layout(width="50%", padding="4px 12px 4px 4px"),
    )
    _col_cam = ipw.VBox(
        [ipw.HTML("<b style='font-size:13px'>Camera geometry</b>",
                  layout=ipw.Layout(margin="0 0 4px 0")),
         s_dd, s_xcen, s_ycen, s_xbet, s_xgam],
        layout=ipw.Layout(width="50%", padding="4px 4px 4px 12px"),
    )

    _controls = ipw.VBox([
        ipw.HBox([_col_orient, _col_cam],
                 layout=ipw.Layout(width="100%")),
        ipw.HBox(
            [btn_reset, btn_setref, btn_accept, btn_fit],
            layout=ipw.Layout(margin="8px 0 6px 0", gap="6px"),
        ),
        ipw.HBox(
            [ipw.HTML("<b>Center at hkl:</b>",
                      layout=ipw.Layout(align_self="center", margin="0 10px 0 0"))]
            + _center_btns,
            layout=ipw.Layout(gap="4px"),
        ),
    ], layout=ipw.Layout(width="100%", padding="4px 8px"))

    _ipy_display(ipw.VBox([fig.canvas, _controls]))

    return state
