import time
import logging
import numpy as np
import matplotlib.pyplot as plt

from core.logger import setup_logging
from core.config import COM_PORT,START_FREQ, STOP_FREQ, POINTS, BULK_READ, CAL_FILE
from acquisition.litevna import LiteVNA
from rf.sparameters import SParameterConverter
from rf.calibration import Calibration

# ============================================================
# LOGGING
# ============================================================
setup_logging()
logger = logging.getLogger("live_view")

# ============================================================
# CONFIG
# ============================================================

FREQ_AXIS = np.linspace(
    START_FREQ,
    STOP_FREQ,
    POINTS
)

# ============================================================
# HELPERS
# ============================================================

def to_db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-12))

# ============================================================
# INIT ACQUISITION
# ============================================================

logger.info("Initializing LiteVNA")

vna = LiteVNA(
    port=COM_PORT,
    start_freq=START_FREQ,
    stop_freq=STOP_FREQ,
    points=POINTS,
    read_size=BULK_READ
)

# ============================================================
# INIT RF PIPELINE
# ============================================================

converter = SParameterConverter()

logger.info("Loading calibration")

cal = Calibration()
cal.load(CAL_FILE)
cal.compute_error_terms()

logger.info("Calibration ready")

# ============================================================
# PLOT SETUP
# ============================================================

plt.ion()

fig, ax = plt.subplots(figsize=(12, 6))

# raw
line_s11_raw, = ax.plot(
    FREQ_AXIS / 1e9,
    np.zeros(POINTS),
    "--",
    label="S11 raw"
)

line_s21_raw, = ax.plot(
    FREQ_AXIS / 1e9,
    np.zeros(POINTS),
    "--",
    label="S21 raw"
)

# calibrated
line_s11_cal, = ax.plot(
    FREQ_AXIS / 1e9,
    np.zeros(POINTS),
    label="S11 calibrated"
)

line_s21_cal, = ax.plot(
    FREQ_AXIS / 1e9,
    np.zeros(POINTS),
    label="S21 calibrated"
)

ax.set_xlabel("Frequency (GHz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title("LiteVNA Live View")

ax.set_ylim(-80, 10)

ax.grid(True)
ax.legend()

plt.show(block=False)
plt.pause(0.1)

background = fig.canvas.copy_from_bbox(fig.bbox)


# ============================================================
# STREAMING LOOP
# ============================================================

logger.info("Starting live stream")
try:
        
    while True:
        loop_t0 = time.perf_counter()
        sweep = vna.acquire_sweep()

        # --------------------------------------------------------
        # RAW RF CONVERSION
        # --------------------------------------------------------
        raw_sparams = converter.from_sweep(sweep)

        # --------------------------------------------------------
        # CALIBRATION
        # --------------------------------------------------------
        cal_sparams = cal.apply(raw_sparams)
        
        # --------------------------------------------------------
        # DISPLAY DATA
        # --------------------------------------------------------

        s11_raw_db = to_db(raw_sparams.s11)
        s21_raw_db = to_db(raw_sparams.s21)

        s11_cal_db = to_db(cal_sparams.s11)
        s21_cal_db = to_db(cal_sparams.s21)

        # --------------------------------------------------------
        # UPDATE PLOTS
        # --------------------------------------------------------

        line_s11_raw.set_ydata(s11_raw_db)
        line_s21_raw.set_ydata(s21_raw_db)

        line_s11_cal.set_ydata(s11_cal_db)
        line_s21_cal.set_ydata(s21_cal_db)

        fig.canvas.restore_region(background)

        ax.draw_artist(line_s11_raw)
        ax.draw_artist(line_s21_raw)

        ax.draw_artist(line_s11_cal)
        ax.draw_artist(line_s21_cal)

        fig.canvas.blit(fig.bbox)
        fig.canvas.flush_events()

        # --------------------------------------------------------
        # PERF LOGGING
        # --------------------------------------------------------

        loop_ms = (time.perf_counter() - loop_t0) * 1000

        logger.info(
            f"Live loop: {loop_ms:7.2f} ms"
        )
except KeyboardInterrupt:
    logger.info("Shutting down...")
except Exception as e:
    logger.error(f"Fatal error: {e}", exc_info=True)