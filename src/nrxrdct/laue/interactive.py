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

_DEG = np.pi / 180


def _gnomonic(tth_deg, chi_deg):
    """Gnomonic projection following the LaueTools convention.

    Projects the scattering-vector direction onto the plane tangent at
    (lat₀ = 45°, long₀ = 0°), which corresponds to 2θ = 90°, χ = 0°.
    Zone axes appear as straight lines in this projection.

    Parameters
    ----------
    tth_deg, chi_deg : array-like  —  2θ and χ in degrees.

    Returns
    -------
    gX, gY : ndarray  —  gnomonic coordinates.
    """
    theta = np.asarray(tth_deg, float) / 2.0
    chi   = np.asarray(chi_deg, float)
    tan_t = np.tan(theta * _DEG)
    safe_tan = np.where(np.abs(tan_t) > 1e-10, tan_t, np.sign(tan_t + 1e-30) * 1e-10)
    lat    = np.arcsin(np.clip(np.cos(theta * _DEG) * np.cos(chi * _DEG), -1.0, 1.0))
    longit = np.arctan(-np.sin(chi * _DEG) / safe_tan)
    # Projection centre at lat0 = π/4 (= 2θ = 90°), longit0 = 0
    lat0, longit0 = np.pi / 4.0, 0.0
    slat0, clat0  = np.sin(lat0), np.cos(lat0)
    slat,  clat   = np.sin(lat),  np.cos(lat)
    cosad  = slat * slat0 + clat * clat0 * np.cos(longit - longit0)
    safe_c = np.where(np.abs(cosad) > 1e-12, cosad, np.nan)
    gX = clat * np.sin(longit0 - longit) / safe_c
    gY = (slat * clat0 - clat * slat0 * np.cos(longit - longit0)) / safe_c
    return gX, gY


def _gnomonic_inv(gX: float, gY: float) -> np.ndarray:
    """Exact inverse of :func:`_gnomonic` — returns unit kf (LT lab frame).

    Inverts the geographic gnomonic centred at lat₀=π/4 (2θ=90°, χ=0°).
    Used by the drag handler to convert a drop position back to a scattering
    direction so the correct U rotation can be computed.
    """
    r = float(np.sqrt(gX ** 2 + gY ** 2))
    if r < 1e-12:
        # Centre of projection: 2θ=90°, χ=0° → kf = (0, 0, 1)
        return np.array([0.0, 0.0, 1.0])

    c     = np.arctan(r)
    sin_c = np.sin(c)
    cos_c = np.cos(c)

    # Invert gnomonic with centre (lat0=π/4, long0=0), sin=cos=1/√2
    lat    = np.arcsin(np.clip((cos_c + gY * sin_c / r) / np.sqrt(2.0), -1.0, 1.0))
    # Note the negation of gX: _gnomonic uses sin(longit0 - longit) = -sin(longit),
    # so gX = -cos(lat)*sin(longit)/cosad — opposite sign from the standard formula.
    longit = np.arctan2(-np.sqrt(2.0) * gX * sin_c, r * cos_c - gY * sin_c)

    cl, sl   = np.cos(lat),    np.sin(lat)
    clo, slo = np.cos(longit), np.sin(longit)

    # Reconstruct kf from (lat, longit) using the forward chain in _gnomonic:
    #   sin(theta) = cos(lat)*cos(longit)
    #   cos(chi)   = sin(lat)/cos(theta)
    #   sin(chi)   = -tan(longit)*tan(theta)
    kfx = 1.0 - 2.0 * cl ** 2 * clo ** 2
    kfy = -2.0 * cl ** 2 * clo * slo
    kfz = 2.0 * sl * cl * clo

    kf = np.array([kfx, kfy, kfz])
    n  = np.linalg.norm(kf)
    return kf / n if n > 1e-12 else kf


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
    space: str = "angular",
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
    space : ``'angular'``, ``'gnomonic'``, or ``'detector'``
        Coordinate frame for the main panel.  ``'angular'`` (default) plots
        2θ (x) vs χ (y) in degrees.  ``'gnomonic'`` plots the gnomonic
        projection k_y/k_x vs k_z/k_x — zone axes appear as straight lines,
        which helps identify multiple grains.  ``'detector'`` plots raw pixel
        positions.  Matching is always done in detector (pixel) space.

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

    ax_det.set_title(
        "Laue — interactive orientation   "
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
    elif space == "gnomonic":
        ax_det.set_aspect("equal")
        ax_det.set_xlabel("gnomonic X", color=_FG, fontsize=8)
        ax_det.set_ylabel("gnomonic Y", color=_FG, fontsize=8)
        ax_det.grid(True, ls=":", lw=0.35, color="#181e2e", zorder=0)
        ax_det.axhline(0, color=_GRAY, lw=0.5, zorder=1)
        ax_det.axvline(0, color=_GRAY, lw=0.5, zorder=1)
    else:  # angular
        ax_det.set_aspect("auto")
        ax_det.set_xlabel("2θ  (°)", color=_FG, fontsize=8)
        ax_det.set_ylabel("χ  (°)", color=_FG, fontsize=8)
        ax_det.grid(True, ls=":", lw=0.35, color="#181e2e", zorder=0)

    # Precompute angular and gnomonic obs coords once — camera is fixed.
    _uf0 = camera.pixel_to_kf(obs_use[:, 0], obs_use[:, 1])
    _obs_angular = np.column_stack([
        np.degrees(np.arccos(np.clip(_uf0[:, 0], -1.0, 1.0))),
        np.degrees(np.arctan2(_uf0[:, 1], _uf0[:, 2] + 1e-17)),
    ])
    _tth0 = np.degrees(np.arccos(np.clip(_uf0[:, 0], -1.0, 1.0)))
    _chi0 = np.degrees(np.arctan2(_uf0[:, 1], _uf0[:, 2] + 1e-17))
    _gX0, _gY0 = _gnomonic(_tth0, _chi0)
    _obs_gnomonic = np.column_stack([_gX0, _gY0])
    _obs_init = (
        _obs_angular  if space == "angular"  else
        _obs_gnomonic if space == "gnomonic" else
        obs_use
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
    sc_sel = ax_det.scatter(                          # selected-spot ring
        [], [], s=130, c="none", edgecolors="#ffff00",
        linewidths=1.8, zorder=7, marker="o",
    )
    sc_drag = ax_det.scatter(                         # drag target ring
        [], [], s=180, c="none", edgecolors="#ff4400",
        linewidths=1.8, zorder=8, marker="o",
    )
    _drag_line, = ax_det.plot(                        # guide line during drag
        [], [], color="#ff4400", lw=1.0,
        linestyle="--", zorder=7, alpha=0.7,
    )
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
        style={"description_width": "130px"},
        layout=ipw.Layout(width="98%"),
    )
    s_ca = ipw.FloatSlider(value=0.0, min=-rot_range_deg,   max=+rot_range_deg,
                           description="Cry [100]  (a)", **_sk)
    s_cb = ipw.FloatSlider(value=0.0, min=-rot_range_deg,   max=+rot_range_deg,
                           description="Cry [010]  (b)", **_sk)
    s_cc = ipw.FloatSlider(value=0.0, min=-c_rot_range_deg, max=+c_rot_range_deg,
                           description="Cry [001]  (c)", **_sk)
    s_hkl = ipw.FloatSlider(value=0.0, min=-rot_range_deg, max=+rot_range_deg,
                             description="rot. [—]", disabled=True, **_sk)

    # State shared between _do_update and the click handler
    _selected: list = [None]   # dict with keys: hkl, tth, chi
    _last     = {"disp": np.empty((0, 2)), "on_det": []}

    _hkl_html = ipw.HTML(
        "<span style='color:#666;font-style:italic'>click a simulated spot to select it</span>",
        layout=ipw.Layout(margin="0 0 2px 4px"),
    )

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

        # Rotation around the selected crystal-plane normal
        if _selected[0] is not None and s_hkl.value != 0.0:
            h, k, l = _selected[0]["hkl"]
            _xtal  = crystal.all_layers[0].crystal if _is_stack else crystal
            G_cry  = _xtal.Q(h, k, l)
            g_norm = np.linalg.norm(G_cry)
            if g_norm > 1e-12:
                ax_hkl = U @ (G_cry / g_norm)
                ax_hkl /= np.linalg.norm(ax_hkl)
                U = Rotation.from_rotvec(
                    np.radians(s_hkl.value) * ax_hkl
                ).as_matrix() @ U

        state.U = U

        dR = U @ np.linalg.inv(state.U0)
        dw = float(np.degrees(Rotation.from_matrix(dR).magnitude()))

        spots  = _simulate(U)
        on_det = [s for s in spots if s.get("pix") is not None][:top_n_sim]
        sim_xy = np.array([s["pix"] for s in on_det]) if on_det else np.empty((0, 2))

        # Build display coordinates
        if space == "angular":
            obs_disp = _obs_angular
            sim_disp = (
                np.array([[s["tth"], s["chi"]] for s in on_det])
                if on_det else np.empty((0, 2))
            )
        elif space == "gnomonic":
            obs_disp = _obs_gnomonic
            if on_det:
                gX, gY   = _gnomonic([s["tth"] for s in on_det], [s["chi"] for s in on_det])
                sim_disp = np.column_stack([gX, gY])
            else:
                sim_disp = np.empty((0, 2))
        else:
            obs_disp = obs_use
            sim_disp = sim_xy

        # Save for click handler
        _last["disp"]   = sim_disp
        _last["on_det"] = on_det

        sc_obs.set_offsets(obs_disp)
        sc_sim.set_offsets(sim_disp if len(sim_disp) else np.empty((0, 2)))

        # Keep selected-spot marker in sync
        sel_pos = np.empty((0, 2))
        if _selected[0] is not None and len(sim_disp):
            sel_hkl = _selected[0]["hkl"]
            for i, s in enumerate(on_det):
                if s.get("hkl") == sel_hkl:
                    sel_pos = sim_disp[i:i + 1]
                    break
        sc_sel.set_offsets(sel_pos)

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

        euler = Rotation.from_matrix(U).as_euler("ZXZ", degrees=True)
        rms_s = f"{rms_px:.1f} px" if np.isfinite(rms_px) else "—"
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
        s_ca.value  = 0.0
        s_cb.value  = 0.0
        s_cc.value  = 0.0
        s_hkl.value = 0.0
        _updating[0] = False
        _do_update()

    def _on_slider(change) -> None:
        if not _updating[0]:
            _do_update()

    for s in (s_ca, s_cb, s_cc, s_hkl):
        s.observe(_on_slider, names="value")

    # ── Click / drag handlers ─────────────────────────────────────────────────
    # A short click selects a simulated spot (enables the HKL rotation slider).
    # Pressing and dragging a spot rotates U so that spot moves to the drop
    # position — a quick way to bootstrap the orientation manually.
    #
    # Drag detection uses screen-pixel distance (event.x / event.y) so the
    # threshold is independent of the current zoom level and display space.

    _drag = {
        "active":       False,
        "spot_idx":     None,
        "orig_disp":    None,   # display coords of spot at press time
        "press_screen": None,   # screen coords (px) at press time
    }

    def _display_to_kf(x: float, y: float) -> np.ndarray:
        """Return unit kf vector (LT lab frame) for a display-space point."""
        if space == "angular":
            tth = np.radians(x)
            chi = np.radians(y)
            return np.array([
                np.cos(tth),
                np.sin(tth) * np.sin(chi),
                np.sin(tth) * np.cos(chi),
            ])
        elif space == "gnomonic":
            return _gnomonic_inv(x, y)
        else:  # detector
            kf = camera.pixel_to_kf(np.array([x]), np.array([y]))[0]
            return kf / np.linalg.norm(kf)

    def _select_spot(idx: int) -> None:
        """Mark spot *idx* as selected and enable the HKL slider."""
        od   = _last["on_det"]
        spot = od[idx]
        _selected[0] = spot
        h, k, l = spot["hkl"]
        _hkl_html.value = (
            f"<b style='color:#ffff00'>({h:+d} {k:+d} {l:+d})</b>"
            f"&nbsp;&nbsp;2θ&nbsp;=&nbsp;{spot['tth']:.2f}°"
            f"&nbsp;&nbsp;χ&nbsp;=&nbsp;{spot['chi']:.2f}°"
        )
        s_hkl.description = f"rot. [{h} {k} {l}]"
        s_hkl.disabled    = False
        _updating[0] = True
        s_hkl.value  = 0.0
        _updating[0] = False
        _do_update()

    def _nearest_sim_idx(xdata: float, ydata: float) -> int | None:
        disp = _last["disp"]
        if len(disp) == 0:
            return None
        dx = disp[:, 0] - xdata
        dy = disp[:, 1] - ydata
        return int(np.argmin(dx ** 2 + dy ** 2))

    def _on_press(event) -> None:
        if event.inaxes is not ax_det or event.button != 1:
            return
        idx = _nearest_sim_idx(event.xdata, event.ydata)
        if idx is None:
            return
        _drag["spot_idx"]     = idx
        _drag["orig_disp"]    = _last["disp"][idx].copy()
        _drag["press_screen"] = np.array([event.x, event.y])
        _drag["active"]       = False

    def _on_motion(event) -> None:
        if event.inaxes is not ax_det or _drag["spot_idx"] is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        # Activate drag only after moving 6 screen pixels — avoids accidental
        # drags on a plain click.
        screen_dist = np.linalg.norm(
            np.array([event.x, event.y]) - _drag["press_screen"]
        )
        if screen_dist < 6.0:
            return
        _drag["active"] = True
        orig = _drag["orig_disp"]
        cur  = np.array([event.xdata, event.ydata])
        sc_drag.set_offsets([cur])
        _drag_line.set_data([orig[0], cur[0]], [orig[1], cur[1]])
        fig.canvas.draw_idle()

    def _on_release(event) -> None:
        if event.button != 1 or _drag["spot_idx"] is None:
            _drag["spot_idx"] = None
            return

        if not _drag["active"]:
            # Short click → select the spot
            _select_spot(_drag["spot_idx"])
        else:
            # Drag released → rotate U so the dragged spot lands at drop point
            if event.xdata is not None and event.ydata is not None:
                orig = _drag["orig_disp"]
                drop = np.array([event.xdata, event.ydata])
                try:
                    ki        = np.array([1.0, 0.0, 0.0])
                    q_orig    = _display_to_kf(orig[0], orig[1]) - ki
                    q_target  = _display_to_kf(drop[0],  drop[1])  - ki
                    n_o, n_t  = np.linalg.norm(q_orig), np.linalg.norm(q_target)
                    if n_o > 1e-10 and n_t > 1e-10:
                        q_o_hat = q_orig   / n_o
                        q_t_hat = q_target / n_t
                        axis    = np.cross(q_o_hat, q_t_hat)
                        sin_a   = np.linalg.norm(axis)
                        cos_a   = float(np.clip(
                            np.dot(q_o_hat, q_t_hat), -1.0, 1.0
                        ))
                        if sin_a > 1e-10:
                            R = Rotation.from_rotvec(
                                np.arctan2(sin_a, cos_a) * axis / sin_a
                            ).as_matrix()
                            state.U0 = R @ state.U
                            _reset_sliders()   # resets sliders and redraws
                except Exception:
                    pass

        # Clear drag visuals
        sc_drag.set_offsets(np.empty((0, 2)))
        _drag_line.set_data([], [])
        _drag["spot_idx"] = None
        _drag["active"]   = False
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event",   _on_press)
    fig.canvas.mpl_connect("motion_notify_event",  _on_motion)
    fig.canvas.mpl_connect("button_release_event", _on_release)

    # ── ipywidgets buttons ────────────────────────────────────────────────────
    _bkw = dict(layout=ipw.Layout(width="145px", height="32px"))
    btn_reset  = ipw.Button(description="Reset",             button_style="warning", **_bkw)
    btn_setref = ipw.Button(description="Set as reference",  button_style="info",    **_bkw)
    btn_accept = ipw.Button(description="✓  Accept",         button_style="success", **_bkw)

    def _cb_reset(_b) -> None:
        state.U0 = state._U0_orig.copy()
        _reset_sliders()

    def _cb_setref(_b) -> None:
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
    btn_setref.on_click(_cb_setref)
    btn_accept.on_click(_cb_accept)

    # ── Quick-fit button ──────────────────────────────────────────────────────
    btn_fit = ipw.Button(description="⚡ Quick Fit", button_style="primary", **_bkw)

    def _cb_fit(_b) -> None:
        import asyncio
        import queue as _qmod
        import threading
        from .fitting import fit_orientation as _fit_orientation

        # Guard against double-click while a fit is already running.
        if getattr(_cb_fit, "_running", False):
            return
        _cb_fit._running = True
        btn_fit.disabled    = True
        btn_fit.description = "Fitting…"

        U_start = state.U.copy()
        _q: _qmod.Queue = _qmod.Queue()

        def _run() -> None:
            try:
                res = _fit_orientation(
                    crystal, camera, obs_use, U_start,
                    max_match_px=max_match_px,
                    top_n_sim=top_n_sim,
                    geometry_only=True,
                    max_nfev=400,
                    verbose=True,
                )
                _q.put(res)
            except Exception as exc:
                _q.put(exc)

        async def _wait() -> None:
            threading.Thread(target=_run, daemon=True).start()
            while _q.empty():
                await asyncio.sleep(0.2)
            item = _q.get_nowait()
            if isinstance(item, Exception):
                print(f"  ⚡ Quick fit error: {item}")
            else:
                # Update state only once, after the fit completes cleanly.
                state.U0 = item.U.copy()
                _reset_sliders()
                print(f"  ⚡ {item}")
            btn_fit.description = "⚡ Quick Fit"
            btn_fit.disabled    = False
            _cb_fit._running    = False

        # create_task is more reliable than ensure_future when the event
        # loop is already running (standard Jupyter / ipykernel setup).
        try:
            asyncio.get_event_loop().create_task(_wait())
        except RuntimeError:
            asyncio.ensure_future(_wait())

    btn_fit.on_click(_cb_fit)

    # ── Save U button ─────────────────────────────────────────────────────────
    btn_save = ipw.Button(description="💾 Save U", button_style="", **_bkw)

    def _cb_save(_b) -> None:
        import glob
        import os
        import re
        existing = glob.glob(os.path.join(os.getcwd(), "UB[0-9]*.npy"))
        max_n = -1
        for f in existing:
            m = re.search(r"UB(\d+)\.npy$", os.path.basename(f))
            if m:
                max_n = max(max_n, int(m.group(1)))
        fname = f"UB{max_n + 1:02d}.npy"
        np.save(fname, state.U)
        print(f"  💾 Saved U → {os.path.abspath(fname)}")

    btn_save.on_click(_cb_save)

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
        s_hkl,
        _hkl_html,
        ipw.HBox(
            [btn_reset, btn_setref, btn_accept, btn_fit, btn_save],
            layout=ipw.Layout(margin="6px 0 6px 0", gap="6px"),
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
    space : ``'angular'``, ``'gnomonic'``, or ``'detector'``
        Coordinate frame for the main panel.  ``'angular'`` (default) plots
        2θ (x) vs χ (y) in degrees.  ``'gnomonic'`` plots the gnomonic
        projection k_y/k_x vs k_z/k_x — zone axes appear as straight lines.
        ``'detector'`` plots raw pixel positions.
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
    elif space == "gnomonic":
        ax_det.set_aspect("equal")
        ax_det.set_xlabel("gnomonic X", color=_FG, fontsize=8)
        ax_det.set_ylabel("gnomonic Y", color=_FG, fontsize=8)
        ax_det.grid(True, ls=":", lw=0.35, color="#181e2e", zorder=0)
        ax_det.axhline(0, color=_GRAY, lw=0.5, zorder=1)
        ax_det.axvline(0, color=_GRAY, lw=0.5, zorder=1)
    else:  # angular
        ax_det.set_aspect("auto")
        ax_det.set_xlabel("2θ  (°)", color=_FG, fontsize=8)
        ax_det.set_ylabel("χ  (°)", color=_FG, fontsize=8)
        ax_det.grid(True, ls=":", lw=0.35, color="#181e2e", zorder=0)

    # Initial observed scatter; angular/gnomonic coords update each frame in
    # _do_update since the camera changes with slider movement.
    _uf0 = camera.pixel_to_kf(obs_use[:, 0], obs_use[:, 1])
    if space == "detector":
        _obs_init = obs_use
    elif space == "gnomonic":
        _tth0_c    = np.degrees(np.arccos(np.clip(_uf0[:, 0], -1.0, 1.0)))
        _chi0_c    = np.degrees(np.arctan2(_uf0[:, 1], _uf0[:, 2] + 1e-17))
        _gX0c, _gY0c = _gnomonic(_tth0_c, _chi0_c)
        _obs_init  = np.column_stack([_gX0c, _gY0c])
    else:  # angular
        _obs_init = np.column_stack([
            np.degrees(np.arccos(np.clip(_uf0[:, 0], -1.0, 1.0))),
            np.degrees(np.arctan2(_uf0[:, 1], _uf0[:, 2] + 1e-17)),
        ])
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
    sc_sel = ax_det.scatter(                          # selected-spot ring
        [], [], s=130, c="none", edgecolors="#ffff00",
        linewidths=1.8, zorder=7, marker="o",
    )
    sc_drag = ax_det.scatter(                         # drag target ring
        [], [], s=180, c="none", edgecolors="#ff4400",
        linewidths=1.8, zorder=8, marker="o",
    )
    _drag_line, = ax_det.plot(                        # guide line during drag
        [], [], color="#ff4400", lw=1.0,
        linestyle="--", zorder=7, alpha=0.7,
    )
    ax_det.legend(loc="upper right", fontsize=7,
                  facecolor=_BG2, edgecolor=_GRAY, labelcolor=_FG)

    ax_info.set_axis_off()
    _info_txt = ax_info.text(
        0.06, 0.98, "",
        transform=ax_info.transAxes,
        color=_FG, fontsize=8.0, va="top", family="monospace",
        linespacing=1.5,
    )

    _last = {"disp": np.empty((0, 2)), "on_det": []}

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
        on_det = [s for s in spots if s.get("pix") is not None][:top_n_sim]

        # Build display coordinates (angular, gnomonic, or detector)
        if space == "angular":
            uf = cam.pixel_to_kf(obs_use[:, 0], obs_use[:, 1])
            tth_o = np.degrees(np.arccos(np.clip(uf[:, 0], -1.0, 1.0)))
            chi_o = np.degrees(np.arctan2(uf[:, 1], uf[:, 2] + 1e-17))
            obs_disp = np.column_stack([tth_o, chi_o])
            sim_disp = (
                np.array([[s["tth"], s["chi"]] for s in on_det])
                if on_det else np.empty((0, 2))
            )
        elif space == "gnomonic":
            uf         = cam.pixel_to_kf(obs_use[:, 0], obs_use[:, 1])
            _tth_o     = np.degrees(np.arccos(np.clip(uf[:, 0], -1.0, 1.0)))
            _chi_o     = np.degrees(np.arctan2(uf[:, 1], uf[:, 2] + 1e-17))
            gX_o, gY_o = _gnomonic(_tth_o, _chi_o)
            obs_disp   = np.column_stack([gX_o, gY_o])
            if on_det:
                gX, gY   = _gnomonic([s["tth"] for s in on_det], [s["chi"] for s in on_det])
                sim_disp = np.column_stack([gX, gY])
            else:
                sim_disp = np.empty((0, 2))
        else:
            obs_disp = obs_use
            sim_disp = sim_xy

        _last["disp"]   = sim_disp
        _last["on_det"] = on_det

        sc_obs.set_offsets(obs_disp)

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

    # ── Click / drag handlers ─────────────────────────────────────────────────
    # Pressing and dragging a simulated spot rotates U so that spot moves to the
    # drop position. Uses state.camera for detector-space kf so the mapping is
    # consistent with the live (slider-adjusted) view.

    _drag = {
        "active":       False,
        "spot_idx":     None,
        "orig_disp":    None,
        "press_screen": None,
    }

    def _display_to_kf_cal(x: float, y: float) -> np.ndarray:
        if space == "angular":
            tth = np.radians(x)
            chi = np.radians(y)
            return np.array([
                np.cos(tth),
                np.sin(tth) * np.sin(chi),
                np.sin(tth) * np.cos(chi),
            ])
        elif space == "gnomonic":
            return _gnomonic_inv(x, y)
        else:  # detector — use current camera to match the live view
            kf = state.camera.pixel_to_kf(np.array([x]), np.array([y]))[0]
            return kf / np.linalg.norm(kf)

    def _nearest_sim_idx_cal(xdata: float, ydata: float) -> int | None:
        disp = _last["disp"]
        if len(disp) == 0:
            return None
        dx = disp[:, 0] - xdata
        dy = disp[:, 1] - ydata
        return int(np.argmin(dx ** 2 + dy ** 2))

    def _on_press(event) -> None:
        if event.inaxes is not ax_det or event.button != 1:
            return
        idx = _nearest_sim_idx_cal(event.xdata, event.ydata)
        if idx is None:
            return
        _drag["spot_idx"]     = idx
        _drag["orig_disp"]    = _last["disp"][idx].copy()
        _drag["press_screen"] = np.array([event.x, event.y])
        _drag["active"]       = False

    def _on_motion(event) -> None:
        if event.inaxes is not ax_det or _drag["spot_idx"] is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        screen_dist = np.linalg.norm(
            np.array([event.x, event.y]) - _drag["press_screen"]
        )
        if screen_dist < 6.0:
            return
        _drag["active"] = True
        orig = _drag["orig_disp"]
        cur  = np.array([event.xdata, event.ydata])
        sc_drag.set_offsets([cur])
        _drag_line.set_data([orig[0], cur[0]], [orig[1], cur[1]])
        fig.canvas.draw_idle()

    def _on_release(event) -> None:
        if event.button != 1 or _drag["spot_idx"] is None:
            _drag["spot_idx"] = None
            return

        if _drag["active"]:
            if event.xdata is not None and event.ydata is not None:
                orig = _drag["orig_disp"]
                drop = np.array([event.xdata, event.ydata])
                try:
                    ki       = np.array([1.0, 0.0, 0.0])
                    q_orig   = _display_to_kf_cal(orig[0], orig[1]) - ki
                    q_target = _display_to_kf_cal(drop[0],  drop[1]) - ki
                    n_o, n_t = np.linalg.norm(q_orig), np.linalg.norm(q_target)
                    if n_o > 1e-10 and n_t > 1e-10:
                        q_o_hat = q_orig   / n_o
                        q_t_hat = q_target / n_t
                        axis    = np.cross(q_o_hat, q_t_hat)
                        sin_a   = np.linalg.norm(axis)
                        cos_a   = float(np.clip(
                            np.dot(q_o_hat, q_t_hat), -1.0, 1.0
                        ))
                        if sin_a > 1e-10:
                            R = Rotation.from_rotvec(
                                np.arctan2(sin_a, cos_a) * axis / sin_a
                            ).as_matrix()
                            state.U0 = R @ state.U
                            _reset_sliders()
                except Exception:
                    pass

        sc_drag.set_offsets(np.empty((0, 2)))
        _drag_line.set_data([], [])
        _drag["spot_idx"] = None
        _drag["active"]   = False
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event",   _on_press)
    fig.canvas.mpl_connect("motion_notify_event",  _on_motion)
    fig.canvas.mpl_connect("button_release_event", _on_release)

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
