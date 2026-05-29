import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from core.logger import setup_logging
from core.config import COM_PORT, START_FREQ, STOP_FREQ, POINTS, BULK_READ, CAL_FILE, _BASE_DIR
from acquisition.litevna import LiteVNA
from rf.sparameters import SParameterConverter
from rf.calibration import Calibration
from perception.features import MagnitudeFeatureExtractor

# ============================================================
# LOGGING
# ============================================================

setup_logging()
logger = logging.getLogger("data_analysis")

# ============================================================
# CONSTANTS
# ============================================================

# ANSI color codes
RESET = '\033[0m'
BOLD = '\033[1m'
CYAN = '\033[36m'
YELLOW = '\033[33m'
RED = '\033[31m'
GREEN = '\033[32m'


class DataAnalyzer:
    """
    Data analysis module for collecting RMG gesture measurements
    across multiple runs and iterations.
    
    Measures all gestures sequentially, then waits for user input
    before repeating the cycle. Saves all data to a single CSV.
    """
    
    def __init__(self, vna, converter, calibration, extractor, verbose=False):
        """
        Initialize analyzer with required RF components.
        
        Args:
            vna: LiteVNA instance for RF measurements
            converter: SParameterConverter for processing sweeps
            calibration: Calibration instance with applied error terms
            extractor: MagnitudeFeatureExtractor for computing magnitudes
            verbose: Enable debug logging for detailed output
        """
        self.vna = vna
        self.converter = converter
        self.calibration = calibration
        self.extractor = extractor
        self.verbose = verbose
        self.analysis_data = None
    
    def _countdown(self, gesture, pause_time=5):
        """
        Display countdown timer before gesture measurement.
        
        Args:
            gesture: Name of the gesture to prepare for
            pause_time: Seconds to count down from
        """
        print(f"\n{CYAN}{BOLD}=============================={RESET}")
        print(f"{CYAN}{BOLD}Prepare for gesture: {YELLOW}{gesture}{RESET}")
        print(f"{CYAN}{BOLD}=============================={RESET}")
        
        for t in range(pause_time, 0, -1):
            color = RED if t <= 3 else YELLOW
            print(f"{color}Starting in {t} seconds...{RESET}", end="\r")
            time.sleep(1)
        
        print(f"{GREEN}{BOLD}>>> GO! Hold the gesture steady <<<{RESET}")
    
    def _acquire_measurement(self):
        """
        Acquire a single RF measurement.
        
        Returns:
            dict: Contains calibrated 'sparams' (SParameters) and 'features' (FeatureVector)
        """
        sweep = self.vna.acquire_sweep()
        raw_sparams = self.converter.from_sweep(sweep)
        cal_sparams = self.calibration.apply(raw_sparams)
        features = self.extractor.extract(cal_sparams)
        
        return {
            'sparams': cal_sparams,
            'features': features
        }
    
    def analysis_routine(
        self,
        gestures,
        iterations=30,
        pause_time=5,
        output_filename=None
    ):
        """
        Main analysis routine: collect RF measurements for multiple gestures across multiple runs.
        
        For each run: measures each gesture N times, then waits for user input.
        Repeats until user stops. Useful for testing system repeatability over time.
        
        Args:
            gestures: List of gesture names to measure
            iterations: Number of measurements per gesture per run (default: 30)
            pause_time: Countdown seconds before each gesture (default: 5)
            output_filename: CSV filename (default: analysis_data.csv)
            
        Returns:
            pd.DataFrame: Analysis dataset with structure:
                gesture, run, iteration, label, S11 0-199, S21 0-199
        """
        if output_filename is None:
            output_filename = "analysis_data.csv"
        
        records = []
        gesture_to_label = {g: i for i, g in enumerate(gestures)}
        run = 0
        
        logger.info(f"Starting analysis routine: {len(gestures)} gestures, {iterations} iterations per gesture")
        
        while True:
            print(f"\n{BOLD}{GREEN}========== RUN {run + 1} =========={RESET}")
            
            for gesture in gestures:
                self._countdown(gesture, pause_time)
                logger.info(f"Run {run + 1}: {gesture} ({iterations} measurements)")
                
                for i in range(iterations):
                    if self.verbose and i % 5 == 0 and i!=0:
                        logger.info(f"{gesture} iteration {i}/{iterations}")
                        self._countdown(gesture, pause_time-2)
                    
                    measurement = self._acquire_measurement()
                    features = measurement['features']
                    
                    record = {
                        'gesture': gesture,
                        'run': run,
                        'iteration': i,
                        'label': gesture_to_label[gesture],
                    }
                    
                    # Extract S11 and S21 magnitudes from feature vector
                    # Features are concatenated as [S11_0..199, S21_0..199]
                    feature_array = features.x
                    s11_mag = feature_array[:POINTS]
                    s21_mag = feature_array[POINTS:2*POINTS]
                    
                    for j, val in enumerate(s11_mag):
                        record[f'S11_{j}'] = val
                    
                    for j, val in enumerate(s21_mag):
                        record[f'S21_{j}'] = val
                    
                    records.append(record)
            
            run += 1
            
            # Wait for input to continue to next run
            print(f"\n{YELLOW}{BOLD}Press ENTER to start the next run, or 'q' to quit...{RESET}")
            user_input = input().strip().lower()
            
            if user_input == 'q':
                break
        
        # Save to CSV
        self.analysis_data = pd.DataFrame.from_records(records)
        output_path = Path(output_filename)
        self.analysis_data.to_csv(output_path, index=False)
        
        logger.info(f"Analysis complete: {len(records)} samples saved to {output_path}")
        
        return self.analysis_data


def main():
    """
    Main entry point: initialize RF pipeline and run analysis routine.
    """
    
    # ============================================================
    # INIT ACQUISITION
    # ============================================================
    
    logger.info("Initializing LiteVNA")
    
    vna = LiteVNA(
        port=COM_PORT,
        start_freq=START_FREQ,
        stop_freq=STOP_FREQ,
        points=POINTS,
        read_size=BULK_READ,
        verbose=True
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
    # INIT FEATURE EXTRACTOR
    # ============================================================
    
    extractor = MagnitudeFeatureExtractor(use_s11=True, use_s21=True, db=False)
    
    # ============================================================
    # CREATE ANALYZER AND RUN ANALYSIS
    # ============================================================
    
    analyzer = DataAnalyzer(vna, converter, cal, extractor, verbose=True)
    
    # Example gestures - modify as needed
    gestures = ["PalmarPinch", "2FingerPinch", "LateralPinch", "Wrap", "Open", "Relaxed"]
    
    # Collect data: 30 measurements per gesture per run, waits for input, then repeats
    analyzer.analysis_routine(
        gestures=gestures,
        iterations=100,
        pause_time=5,
        output_filename= _BASE_DIR / "output" / "analysis_data.csv"
    )
    
    logger.info("Data analysis complete")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
