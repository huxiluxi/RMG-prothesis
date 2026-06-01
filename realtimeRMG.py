import logging
import time

from core.config import COM_PORT,START_FREQ, STOP_FREQ, POINTS, BULK_READ, CAL_FILE, HAND_JOINT, HAND_TYPE, CAN_INTERFACE, _BASE_DIR
from core.logger import setup_logging

from acquisition.litevna import LiteVNA
from rf.sparameters import SParameterConverter
from rf.calibration import Calibration
from perception.features import MagnitudeFeatureExtractor
from perception.classifier import Classifier
from perception.trainer import Trainer
from decision.commitment import CommitmentFilter

# control
from control.linker_hand import HandMotionPlanner
from control.linkerhand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi


# ============================================================
# LOGGING
# ============================================================

setup_logging()
logger = logging.getLogger("realtimeRMG")

logger.info("Initializing RMGrealtime system")

# ============================================================
# HARDWARE INITIALIZATION
# ============================================================

logger.info("Initializing LiteVNA")
vna = LiteVNA(
    port=COM_PORT,
    start_freq=START_FREQ,
    stop_freq=STOP_FREQ,
    points=POINTS,
    read_size=BULK_READ,
    verbose=False
)

logger.info("Loading calibration")
cal = Calibration()
cal.load(CAL_FILE)
cal.compute_error_terms()

converter = SParameterConverter()

# ============================================================
# ML PIPELINE INITIALIZATION
# ============================================================

logger.info("Initializing ML pipeline")
extractor = MagnitudeFeatureExtractor(verbose=True)
model = Classifier(verbose=True,n_PC=30)
commit = CommitmentFilter(verbose=False,window=8)

trainer = Trainer(vna, converter, cal, extractor, verbose=True)

hand = LinkerHandApi(
    hand_joint=HAND_JOINT,
    hand_type=HAND_TYPE,
    can=CAN_INTERFACE
)
planner = HandMotionPlanner(hand)

# ============================================================
# MAIN LOOP - INFERENCE MODE
# ============================================================

def run_inference(gestures):
    """Run real-time gesture inference."""
    logger.info("Starting inference loop")
    
    while True:
        loop_t0 = time.perf_counter()

        # Acquisition
        t0 = time.perf_counter()
        sweep = vna.acquire_sweep()
        t_acquire = (time.perf_counter() - t0) * 1000

        # S-parameter conversion
        t0 = time.perf_counter()
        raw_sparams = converter.from_sweep(sweep)
        t_convert = (time.perf_counter() - t0) * 1000

        # Calibration
        t0 = time.perf_counter()
        cal_sparams = cal.apply(raw_sparams)
        t_calib = (time.perf_counter() - t0) * 1000

        # Feature extraction
        t0 = time.perf_counter()
        feat = extractor.extract(cal_sparams)
        t_extract = (time.perf_counter() - t0) * 1000

        # Model prediction
        t0 = time.perf_counter()
        pred = model.predict(feat.x)
        t_predict = (time.perf_counter() - t0) * 1000

        # Commitment filter
        t0 = time.perf_counter()
        stable = commit.update(pred)
        t_commit = (time.perf_counter() - t0) * 1000

        stable_text = gestures[stable]
        prediction_text = gestures[pred]

        stable_color = "\033[31m" if stable_text == "No action" else "\033[32m"

        logger.info(
            f"Prediction: \033[33m{prediction_text}\033[0m | "
            f"Stable: {stable_color}\033[1m{stable_text}\033[0m"
        )
        
        if(gestures[stable] != "No action"):
            planner.action(gestures[stable])

        loop_ms = (time.perf_counter() - loop_t0) * 1000

        # logger.info(f"TIMING | Acquire: {t_acquire:7.2f} ms | Convert: {t_convert:7.2f} ms | Calib: {t_calib:7.2f} ms | Extract: {t_extract:7.2f} ms | Predict: {t_predict:7.2f} ms | Commit: {t_commit:7.2f} ms | Total: {loop_ms:7.2f} ms")

# ============================================================
# TRAINING MODE
# ============================================================

def run_training(gestures, iterations=30, pause_time=5, output_filename="Training_data.csv"):
    """
    Run gesture training routine.
    
    Args:
        gestures: List of gesture names to train on
        iterations: Number of measurements per gesture
        pause_time: Countdown time before each gesture
        output_filename: Output CSV filename
    """
    logger.info(f"Starting training mode with gestures: {gestures}")
    
    # Collect training data
    training_data = trainer.training_routine(
        gestures=gestures,
        iterations=iterations,
        pause_time=pause_time,
        output_filename=output_filename
    )
    
    # Fit classifier on collected data
    logger.info("Fitting classifier on training data")
    model.fit(training_data)
    
    logger.info("Training complete")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("RMGrealtime System Started")
    logger.info("="*60)
    
    # Example: Uncomment to run training
    gestures = ["PalmarPinch", "2FingerPinch", "LateralPinch", "Wrap", "Open", "Relaxed"]
    # gestures = ["PalmarPinch", "Open"]
    
    run_training(
        gestures=gestures,
        iterations=60,
        pause_time=5,
        output_filename=_BASE_DIR / "output" / "training_data.csv"
    )
    gestures.append("No action")
    # model.fit_from_file(_BASE_DIR / "output" / "training_data.csv")
    
    # Run inference by default
    try:
        run_inference(gestures)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)