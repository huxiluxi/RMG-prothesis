# apps/calibration_SOLT.py

import logging

from core.logger import setup_logging
from core.config import COM_PORT,START_FREQ, STOP_FREQ, POINTS, BULK_READ, CAL_FILE
from acquisition.litevna import LiteVNA
from rf.sparameters import SParameterConverter
from rf.calibration import Calibration


# ============================================================
# LOGGING
# ============================================================

setup_logging()
logger = logging.getLogger("calibration_SOLT")

# ============================================================
# PROMPT
# ============================================================

def prompt_measurement(name: str):

    logger.info("========================================")
    logger.info(f"Connect: {name}")
    logger.info("Press ENTER to continue")
    logger.info("========================================")

    input()

# ============================================================
# FULL SWEEP
# ============================================================

def acquire_full_sparameters(vna: LiteVNA, converter: SParameterConverter):

    logger.info("Acquiring full sweep")

    sweep = vna.acquire_sweep()
    # print(sweep)

    sparams = converter.from_sweep(sweep)
    # print(sparams)

    logger.info("Sweep complete")

    return sparams

# ============================================================
# MAIN
# ============================================================

def prompt_load_or_new():
    """Prompt user to choose between loading existing calibration or performing a new one."""
    logger.info("========================================")
    logger.info("1. Load existing calibration")
    logger.info("2. Perform new calibration")
    logger.info("========================================")
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ['1', '2']:
            return choice
        logger.warning("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    
    logger.info("Starting SOLT calibration")

    # --------------------------------------------------------
    # CHOOSE: LOAD OR NEW CALIBRATION
    # --------------------------------------------------------

    choice = prompt_load_or_new()

    if choice == '1':
        logger.info("Loading existing calibration")
    else:
        logger.info("Performing new calibration acquisition")

        # --------------------------------------------------------
        # Initialize VNA
        # --------------------------------------------------------

        vna = LiteVNA(
            port=COM_PORT,
            start_freq=START_FREQ,
            stop_freq=STOP_FREQ,
            points=POINTS,
            read_size=BULK_READ
        )

        converter = SParameterConverter()

        # --------------------------------------------------------
        # SHORT
        # --------------------------------------------------------

        prompt_measurement("SHORT on PORT 1")

        short = acquire_full_sparameters(vna,converter)

        # --------------------------------------------------------
        # OPEN
        # --------------------------------------------------------

        prompt_measurement("OPEN on PORT 1")

        open_ = acquire_full_sparameters(vna,converter)

        # --------------------------------------------------------
        # LOAD + ISOLATION
        # --------------------------------------------------------

        prompt_measurement("LOAD on PORT 1 + PORT 2")

        load = acquire_full_sparameters(vna,converter)

        # --------------------------------------------------------
        # THROUGH
        # --------------------------------------------------------

        prompt_measurement("THROUGH between PORT 1 and PORT 2")

        thru = acquire_full_sparameters(vna,converter)

        # --------------------------------------------------------
        # SAVE
        # --------------------------------------------------------

        logger.info(f"Saving calibration to: {CAL_FILE}")

        with open(CAL_FILE, "w") as f:

            f.write("# LiteVNA calibration\n")
            f.write(
                "# Hz "
                "ShortR ShortI "
                "OpenR OpenI "
                "LoadR LoadI "
                "ThroughR ThroughI "
                "IsolationR IsolationI\n"
            )

            for i in range(POINTS):

                freq = int(short.freq[i])

                f.write(
                    f"{freq} "

                    f"{short.s11[i].real} {short.s11[i].imag} "

                    f"{open_.s11[i].real} {open_.s11[i].imag} "

                    f"{load.s11[i].real} {load.s11[i].imag} "

                    f"{thru.s21[i].real} {thru.s21[i].imag} "

                    f"{load.s21[i].real} {load.s21[i].imag}\n"
                )

        logger.info("Calibration saved successfully")

    # --------------------------------------------------------
    # COMPUTE AND PLOT
    # --------------------------------------------------------

    cal = Calibration()
    cal.load(CAL_FILE)
    cal.compute_error_terms()
    cal.plot_error_terms()
    