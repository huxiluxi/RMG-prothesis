import re
import numpy as np
import logging
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict
from scipy.interpolate import interp1d

from core.types import SParameters, CalData
from core.config import _BASE_DIR
from output.export_pubfig import export_pubfig

logger = logging.getLogger("rf.calibration")

# ============================================================
# IDEAL STANDARDS
# ============================================================

IDEAL_SHORT = -1 + 0j
IDEAL_OPEN = 1 + 0j
IDEAL_LOAD = 0 + 0j
IDEAL_THROUGH = 1 + 0j

# ============================================================
# CALIBRATION ENGINE
# ============================================================

class Calibration:

    def __init__(self, verbose=False):
        self.data = defaultdict(CalData)
        self.interp = {}
        self.verbose = verbose
        
        if self.verbose:
            logger.info("Initialized Calibration")

    # ------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------

    def load(self, filename):
        logger.info(f"Loading calibration: {filename}")

        line_re = re.compile(r"""
            ^\s*
            (?P<freq>\d+)\s+
            (?P<shortr>[-0-9Ee.]+)\s+
            (?P<shorti>[-0-9Ee.]+)\s+
            (?P<openr>[-0-9Ee.]+)\s+
            (?P<openi>[-0-9Ee.]+)\s+
            (?P<loadr>[-0-9Ee.]+)\s+
            (?P<loadi>[-0-9Ee.]+)
            (\s+(?P<throughr>[-0-9Ee.]+)\s+(?P<throughi>[-0-9Ee.]+))?
            (\s+(?P<thrureflr>[-0-9Ee.]+)\s+(?P<thrurefli>[-0-9Ee.]+))?
            (\s+(?P<isolationr>[-0-9Ee.]+)\s+(?P<isolationi>[-0-9Ee.]+))?
        """, re.VERBOSE)

        with open(filename, "r") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue

                m = line_re.match(line)
                if not m:
                    continue

                d = m.groupdict()
                freq = float(d["freq"])

                cal = self.data[freq]
                cal.freq = freq

                cal.short = complex(float(d["shortr"]), float(d["shorti"]))
                cal.open = complex(float(d["openr"]), float(d["openi"]))
                cal.load = complex(float(d["loadr"]), float(d["loadi"]))

                if d["throughr"]:
                    cal.through = complex(float(d["throughr"]), float(d["throughi"]))

                if d["thrureflr"]:
                    cal.thrurefl = complex(float(d["thrureflr"]), float(d["thrurefli"]))

                if d["isolationr"]:
                    cal.isolation = complex(float(d["isolationr"]), float(d["isolationi"]))

        logger.info(f"Loaded {len(self.data)} calibration points")
        
        if self.verbose:
            logger.info(f"Calibration points details: {len(self.data)} frequencies loaded")

    # ------------------------------------------------------------
    # ERROR TERMS
    # ------------------------------------------------------------

    def compute_error_terms(self):
        logger.info("Computing calibration error terms")
        
        if self.verbose:
            logger.info(f"Computing error terms for {len(self.data)} calibration points")

        for freq, cal in self.data.items():

            g1, g2, g3, gt = IDEAL_SHORT, IDEAL_OPEN, IDEAL_LOAD, IDEAL_THROUGH
            gm1, gm2, gm3 = cal.short, cal.open, cal.load

            denom = (
                g1 * (g2 - g3) * gm1 +
                g2 * g3 * gm2 -
                g2 * g3 * gm3 -
                (g2 * gm2 - g3 * gm3) * g1
            )

            if denom == 0:
                continue

            cal.e00 = (
                -((g2 * gm3 - g3 * gm3) * g1 * gm2 -
                  (g2 * g3 * gm2 - g2 * g3 * gm3 -
                   (g3 * gm2 - g2 * gm3) * g1) * gm1)
            ) / denom

            cal.e11 = (
                (g2 - g3) * gm1 -
                g1 * (gm2 - gm3) +
                g3 * gm2 -
                g2 * gm3
            ) / denom

            cal.delta_e = -(
                (g1 * (gm2 - gm3) - g2 * gm2 + g3 * gm3) * gm1 +
                (g2 * gm3 - g3 * gm3) * gm2
            ) / denom

            cal.e10e01 = cal.e00 * cal.e11 - cal.delta_e

            if cal.through != 0j:
                gm4 = cal.through
                gm5 = cal.thrurefl
                gm6 = cal.isolation

                gm7 = gm5 - cal.e00

                cal.e30 = gm6

                cal.e22 = gm7 / (gm7 * cal.e11 * gt**2 + cal.e10e01 * gt**2)

                cal.e10e32 = (gm4 - gm6) * (1 - cal.e11 * cal.e22 * gt**2) / gt

        self._build_interp()

    # ------------------------------------------------------------
    # INTERPOLATION
    # ------------------------------------------------------------

    def _build_interp(self):
        freqs = np.array(sorted(self.data.keys()))

        def arr(name):
            return np.array([getattr(self.data[f], name) for f in freqs])

        self.interp = {
            k: interp1d(freqs, arr(k),bounds_error=False ,fill_value="extrapolate")
            for k in [
                "e00", "e11", "delta_e",
                "e10e01", "e30", "e22", "e10e32"
            ]
        }

    # ------------------------------------------------------------
    # CORRECTION
    # ------------------------------------------------------------

    def apply(self, sparams: SParameters) -> SParameters:
        """
        RF-layer compliant calibration:
        SParameters → SParameters
        """

        s11_out = np.zeros_like(sparams.s11)
        s21_out = np.zeros_like(sparams.s21)

        for i, freq in enumerate(sparams.freq):
            e00 = self.interp["e00"](freq)
            e11 = self.interp["e11"](freq)
            de  = self.interp["delta_e"](freq)

            e30 = self.interp["e30"](freq)
            e10e32 = self.interp["e10e32"](freq)
            e10e01 = self.interp["e10e01"](freq)

            s11 = sparams.s11[i]
            s21 = sparams.s21[i]

            # 1-port correction
            s11c = (s11 - e00) / (s11 * e11 - de)

            # 2-port correction
            s21c = (s21 - e30) / e10e32
            s21c *= e10e01 / (e11 * s11 - de)

            s11_out[i] = s11c
            s21_out[i] = s21c

        return SParameters(
            freq=sparams.freq,
            s11=s11_out,
            s21=s21_out
        )

    # ------------------------------------------------------------
    # DEBUG PLOT
    # ------------------------------------------------------------

    def plot_error_terms(self):
        freqs = np.array(sorted(self.data.keys())) / 1e9
        
        # Reflection-related error terms (S11 measurements)
        reflection_terms = ["e00", "e11", "delta_e", "e22"]
        # Transmission-related error terms (S21 measurements)
        transmission_terms = ["e10e01", "e10e32", "e30"]
        
        # Figure 1: Reflection-related error terms (2x2 grid)
        fig1, axes1 = plt.subplots(2, 2, figsize=(12, 9))
        axes1 = axes1.flatten()
        fig1.suptitle("Reflection-Related Error Terms (S11)", fontsize=13, fontweight='bold')
        
        for idx, name in enumerate(reflection_terms):
            vals = np.array([getattr(self.data[f], name) for f in sorted(self.data.keys())])
            
            ax = axes1[idx]
            ax.plot(freqs, 20 * np.log10(np.abs(vals) + 1e-15), label="Magnitude (dB)", linewidth=1.5)
            ax.plot(freqs, np.unwrap(np.angle(vals)) * 180/np.pi, label="Phase (°)", linewidth=1.5)
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Magnitude (dB) / Phase (°)")
            ax.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        
        # Export Figure 1
        export_pubfig(fig1, _BASE_DIR / "output" / "calibration_reflection_error_terms.pdf", width=12, aspectRatio=0.75)
        
        # Figure 2: Transmission-related error terms (centered e30)
        from matplotlib import gridspec
        fig2 = plt.figure(figsize=(12, 9))
        fig2.suptitle("Transmission-Related Error Terms (S21)", fontsize=13, fontweight='bold')
        gs = gridspec.GridSpec(2, 4, figure=fig2)
        
        # Top row: e10e01 and e10e32
        axes2 = [fig2.add_subplot(gs[0, 0:2]), fig2.add_subplot(gs[0, 2:4])]
        # Bottom row: e30 centered
        axes2.append(fig2.add_subplot(gs[1, 1:3]))
        
        for idx, name in enumerate(transmission_terms):
            vals = np.array([getattr(self.data[f], name) for f in sorted(self.data.keys())])
            
            ax = axes2[idx]
            ax.plot(freqs, 20 * np.log10(np.abs(vals) + 1e-15), label="Magnitude (dB)", linewidth=1.5)
            ax.plot(freqs, np.unwrap(np.angle(vals)) * 180/np.pi, label="Phase (°)", linewidth=1.5)
            ax.set_title(name, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("Frequency (GHz)")
            ax.set_ylabel("Magnitude (dB) / Phase (°)")
            ax.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        
        # Export Figure 2
        export_pubfig(fig2, _BASE_DIR / "output" / "calibration_transmission_error_terms.pdf", width=12, aspectRatio=0.75)
        
        plt.show()