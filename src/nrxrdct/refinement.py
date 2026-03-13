import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import GSASIIscriptable as G2sc

from .parameters import Scan
from .io import read_xy_file, write_starting_instrument_pars

class InstrumentCalibration(Scan):

    def __init__(self, acquisition_file, sample_name, scan_type = "half-turn", translation_motor = "dty", rotation_motor = "rot", outer_loop_motor = "translation", beam_size = 0.0001, beam_energy = 44, tth_lims:tuple=(None, None), xy_file:Path=Path("integrated_data.xy"), param_file:Path=Path('calibrated_instrument.instprm'), polarization:float=0.99):
        super().__init__(acquisition_file, sample_name, scan_type, translation_motor, rotation_motor, outer_loop_motor, beam_size, beam_energy)
        
        os.makedirs("calibration", exist_ok=True)

        self.xy_file = xy_file
        self.param_file = param_file
        self.low_lim, self.high_lim = tth_lims
        self.tth, self.intensity = read_xy_file(str(self.xy_file))
        self.phases = []

        if self.low_lim == None:
            self.low_lim = self.tth.min()
        if self.high_lim == None:
            self.hih_lim = self.tth.max()        

        self.param_file_init = write_starting_instrument_pars(polarization=polarization, wavelength=self.wavelength)

    def create_model(self, gpx_file:Path=Path("model.gpx")):

        self.gpx = G2sc.G2Project(newgpx=str(gpx_file))

        self.hist = self.gpx.add_powder_histogram(
            datafile=self.xy_file,
            iparams=self.param_file_init,
            phases="all"
        )
        self.hist["data"][0]["Limits"] = [self.low_lim, self.high_lim]
        self.gpx.save()
        return self.gpx, self.hist
    
    def add_phase(self, cif_file:Path=Path("cif_file"), phase_name:str="LaB6", block_cell:bool=True):
        self.calibrant_composition = phase_name
        self.phase = self.gpx.add_phase(str(cif_file), phasename=phase_name,                     histograms=[self.hist])
        if block_cell:
            for atom in self.phase.atoms():
                atom.refinement_flags = ""
            self.phase.set_refinements({"Cell": False})
        self.phases.append(self.phase)
        self.gpx.save()
        return self.phase
    
    def refine_background(self, number_coeff:int=12, do_refine:bool=True):
        # Step 6.1: Background
        self.hist.set_refinements({"Background": {"no. coeffs": number_coeff, "refine": do_refine}})
        self.gpx.save()
        self.gpx.do_refinements([{}])
        print("Background refinement performed")

    def refine_scale(self):
        self.hist.SampleParameters['Scale'][1] = True
        self.gpx.save()
        self.gpx.do_refinements([{}])
        print("Step 6.2 done - scale")

    def refine_zero_shift(self):
        self.hist.set_refinements({"Instrument Parameters": ["Zero"]})
        self.gpx.save()
        self.gpx.do_refinements([{}])
        print("Step 6.3 done - zero shift")

    def refine_gaussian_broadening(self, refine:list=["U", "V", "W", "SH/L"]):
        """
        Possible parameters are U, V, W, SH/L
        For synchrotron light with 2D detectors, go only for W. 
        """
        for param in refine:
            self.hist.set_refinements({"Instrument Parameters":[param]})
            self.gpx.save()
            self.gpx.do_refinements([{}])
            print(f"Refined {param} (Gaussian broadening)")

    def refine_lorentzian_broadening(self, refine:list=["X", "Y"]):
        """
        Possible parameters are X, and Y
        """
        for param in refine:
            self.hist.set_refinements({"Instrument Parameters":[param]})
            self.gpx.save()
            self.gpx.do_refinements([{}])
            print(f"Refined {param} (Lorentzian broadening)")

    def _refine_cell(self):
        self.phase.set_refinements({"Cell": True})
        self.gpx.save()
        self.gpx.do_refinements([{}])

    def _refine_texture_broadening(self):
        
        self.phase.set_HAP_refinements({"Pref.ori":True, "Model":"MD", "Axis":[1,1,1], "Ratio":1.0, "Ref":True})
        self.gpx.save()
        self.gpx.do_refinements([{}])

    def _refine_atomic_positions(self, flags:list=["X", "XU"]):
        for flag in flags:
            for atom in self.phase.atoms():
                atom.refinement_flags = flag
            self.gpx.save()
            self.gpx.do_refinements([{}])


    def print_refinement_results(self,):
        print("\n" + "=" * 60)
        print("REFINEMENT RESULTS")
        print("=" * 60)

        ip = self.hist["Instrument Parameters"][0]
        params_to_report = ["Lam", "Zero", "U", "V", "W", "X", "Y", "SH/L", "Polariz."]
        calibrated = {}
        for p in params_to_report:
            if p in ip:
                val = ip[p][1] if isinstance(ip[p], list) else ip[p]
                calibrated[p] = val

        print("\nCalibrated instrument parameters:")
        for p, v in calibrated.items():
            fixed = "" if (isinstance(ip[p], list) and ip[p][2]) else " (fixed)"
            print(f"  {p:12s} = {v:.6f}{fixed}")

        wR = self.hist.get_wR()
        print(f"\n  Rwp = {wR:.2f} %" if wR is not None else "\n  Rwp = refinement failed")
        print("\nPhase results:")
        for ph in self.gpx.phases():
            print(f"\n  Phase: {ph.name}")
            cell = ph.get_cell()
            for k, v in cell.items():
                try:
                    print(f"    {k} = {v:.5f}")
                except Exception:
                    print(f"    {k} = {v}")

    def write_calibrated_intrument_pars(self, output_file:Path=Path("calibrated_instrument.instprm")):
        
        save_path = Path("calibration") / output_file

        print("\n" + "=" * 60)
        print("EXPORTING CALIBRATED INSTPRM")
        print("=" * 60)

        ip = self.hist["Instrument Parameters"][0]
        lines = ["#GSAS-II instrument parameter file\n"]

        key_order_single = ["Type", "Bank", "Zero", "Polariz.", "Azimuth",
                            "U", "V", "W", "X", "Y", "Z", "SH/L"]

        lines.append(f"Lam:{self.wavelength}\n")

        for p in key_order_single:
            if p in ip:
                val = ip[p][1] if isinstance(ip[p], list) else ip[p]
                lines.append(f"{p}:{val}\n")

        with open(str(save_path), "w") as f:
            f.writelines(lines)

        print(f"Calibrated instprm saved to:\n  {str(save_path)}")
        print("\nFinal calibrated parameters:")
        for line in lines[1:]:   # skip the header comment
            print(f"  {line.rstrip()}")

        print(f"Calibrated instprm saved to:\n  {str(save_path)}")
        print("\nIn your sample refinements:")
        print("  - Use this file as INST_PARAMS")
        print("  - Fix Zero, W, X, Y (carry from calibration)")
        print("  - U, V, SH/L remain fixed at 0 / 0.0001 for synchrotron and 2D detectors")

    def plot_results(self, image_path:Path="calibration_plot.png"):

        save_path = Path("calibration") / image_path 

        ip = self.hist["Instrument Parameters"][0]
        params_to_report = ["Lam", "Zero", "U", "V", "W", "X", "Y", "SH/L", "Polariz."]
        calibrated = {}
        for p in params_to_report:
            if p in ip:
                val = ip[p][1] if isinstance(ip[p], list) else ip[p]
                calibrated[p] = val

        wR = self.hist.get_wR()
        print("\n" + "=" * 60)
        print("Generating calibration plot")
        print("=" * 60)

        fig = plt.figure(figsize=(15, 10))
        gs  = gridspec.GridSpec(3, 2, figure=fig,
                                height_ratios=[3, 1, 1.2],
                                hspace=0.08, wspace=0.35)

        ax_main = fig.add_subplot(gs[0, 0])
        ax_diff = fig.add_subplot(gs[1, 0], sharex=ax_main)
        ax_ip   = fig.add_subplot(gs[0, 1])
        ax_fw   = fig.add_subplot(gs[1, 1])
        ax_text = fig.add_subplot(gs[2, :])
        ax_text.axis("off")

        # --- Observed / Calculated / Difference ---
        tth   = self.hist.getdata('x')
        yobs  = self.hist.getdata('yobs')
        ycalc = self.hist.getdata('ycalc')
        ybkg  = self.hist.getdata('background')
        diff  = yobs - ycalc

        ax_main.plot(tth, yobs,  'k.',  ms=2,   label='Observed')
        ax_main.plot(tth, ycalc, 'r-',  lw=1,   label='Calculated')
        ax_main.plot(tth, ybkg,  'b--', lw=0.8, label='Background')

        # Reflection tick marks for both phases
        colours_ticks = {'calibrant': 'magenta', 'Al_holder': 'darkorange'}
        yrange = yobs.max() - yobs.min()
        tick_y0 = yobs.min() - 0.02*yrange
        for ph in self.gpx.phases():
            try:
                reflist = self.hist.reflections()[ph.name]['RefList']
                ref_tth = reflist[:, 5]
                colour  = colours_ticks.get(ph.name, 'green')
                ax_main.vlines(ref_tth, tick_y0, tick_y0 + 0.04*yrange,
                            color=colour, lw=0.8, label=f'{ph.name} reflections')
            except Exception:
                pass

        ax_main.set_ylabel('Intensity')
        ax_main.set_title(
            f'{self.calibrant_composition}\nRwp = {wR:.2f} %' if wR else 'Calibration fit')
        ax_main.legend(fontsize=7, markerscale=2)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        ax_diff.plot(tth, diff, color='forestgreen', lw=0.8)
        ax_diff.axhline(0, color='k', lw=0.5, ls='--')
        ax_diff.set_ylabel('Obs−Calc')
        ax_diff.set_xlabel('2θ (degrees)')

        # --- Bar chart of calibrated parameters ---
        bar_params = ["Zero", "W", "X", "Y"]
        bar_vals   = [calibrated.get(p, 0.0) for p in bar_params]
        colours_bar = ['steelblue' if v >= 0 else 'tomato' for v in bar_vals]

        bars = ax_ip.bar(bar_params, bar_vals, color=colours_bar, edgecolor='k', lw=0.6)
        ax_ip.axhline(0, color='k', lw=0.5)
        ax_ip.set_title('Refined parameters\n(U, V, SH/L fixed at 0)')
        ax_ip.set_ylabel('Value')
        spread = max(abs(v) for v in bar_vals) if any(bar_vals) else 1
        for bar, val in zip(bars, bar_vals):
            ax_ip.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01*spread,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7)

        # --- FWHM vs 2theta ---
        try:
            W_v = calibrated.get('W', 0)
            X_v = calibrated.get('X', 0)
            Y_v = calibrated.get('Y', 0)
            tth_range = np.linspace(self.low_lim, self.high_lim, 300)
            tan_th    = np.tan(np.radians(tth_range / 2))
            cos_th    = np.cos(np.radians(tth_range / 2))
            fwhm_G    = np.sqrt(np.abs(W_v)) * np.ones_like(tth_range)
            fwhm_L    = np.abs(X_v) / cos_th + np.abs(Y_v) * tan_th
            fwhm_total= (fwhm_G**5 + fwhm_L**5)**(1/5)
            ax_fw.plot(tth_range, fwhm_G,     'darkorange', lw=1.5, label='Gaussian (W)')
            ax_fw.plot(tth_range, fwhm_L,     'steelblue',  lw=1.5, label='Lorentzian (X+Y)')
            ax_fw.plot(tth_range, fwhm_total, 'k-',         lw=1.5, label='TCH total')
            ax_fw.set_xlabel('2θ (degrees)')
            ax_fw.set_ylabel('FWHM (degrees)')
            ax_fw.set_title('Peak width model')
            ax_fw.legend(fontsize=7)
            ax_fw.grid(True, alpha=0.3)
        except Exception as e:
            ax_fw.text(0.5, 0.5, f'FWHM plot error:\n{e}',
                    ha='center', va='center', transform=ax_fw.transAxes, fontsize=8)

        # --- Parameter table ---
        table_params = ["Lam", "Zero", "U", "V", "W", "X", "Y", "SH/L", "Polariz."]
        table_vals   = [f"{calibrated.get(p, 0.0):.6f}" for p in table_params]
        tbl = ax_text.table(
            cellText=[table_vals],
            colLabels=table_params,
            loc='center', cellLoc='center'
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.scale(1, 1.8)
        ax_text.set_title('Refined instrument parameter values\n'
                        '(U=V=0, SH/L=0.0001 fixed — not meaningful for 2D detector)',
                        pad=4, fontsize=9)

        plt.suptitle('GSAS-II Instrument Calibration',
                    fontsize=12, fontweight='bold', y=1.01)
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
        plt.show()


