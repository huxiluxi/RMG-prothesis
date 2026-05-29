import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from core.types import SParameters, FeatureVector

logger = logging.getLogger(__name__)


class Trainer:
    """
    Gesture training module for collecting calibrated RF measurements
    across multiple gestures and iterations.
    
    Collects raw S-parameters and features, then saves to CSV for model training.
    """
    
    # ANSI color codes
    RESET = '\033[0m'
    BOLD = '\033[1m'
    CYAN = '\033[36m'
    YELLOW = '\033[33m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    
    def __init__(self, vna, converter, calibration, extractor, verbose=False):
        """
        Initialize trainer with required RF components.
        
        Args:
            vna: LiteVNA instance for RF measurements
            converter: SParameterConverter for processing sweeps
            calibration: Calibration instance with applied error terms
            extractor: MagnitudeFeatureExtractor for computing magnitudes
            verbose: Enable debug logging for detailed step-by-step output
        """
        self.vna = vna
        self.converter = converter
        self.calibration = calibration
        self.extractor = extractor
        
        self.verbose = verbose

        self.training_data = None
    
    def _countdown(self, gesture, pause_time=5):
        """
        Display countdown timer before gesture measurement.
        
        Args:
            gesture: Name of the gesture to prepare for
            pause_time: Seconds to count down from
        """
        print(f"\n{self.CYAN}{self.BOLD}=============================={self.RESET}")
        print(f"{self.CYAN}{self.BOLD}Prepare for gesture: {self.YELLOW}{gesture}{self.RESET}")
        print(f"{self.CYAN}{self.BOLD}=============================={self.RESET}")
        
        for t in range(pause_time, 0, -1):
            color = self.RED if t <= 3 else self.YELLOW
            print(f"{color}Starting in {t} seconds...{self.RESET}", end="\r")
            time.sleep(1)
        
        print(f"{self.GREEN}{self.BOLD}>>> GO! Hold the gesture steady <<<{self.RESET}")
    
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
        
    def training_routine(self, gestures, iterations=10, pause_time=5, output_filename=None):
        """
        Main training routine: collect RF measurements for multiple gestures.
        
        Args:
            gestures: List of gesture names to measure
            iterations: Number of measurements per gesture
            pause_time: Countdown seconds before each gesture (default: 5)
            output_filename: CSV filename (default: Training_data.csv)
            
        Returns:
            pd.DataFrame: Training dataset with structure:
                gesture, iteration, label, S11 0-199, S21 0-199
        """
        if output_filename is None:
            output_filename = "Training_data.csv"
        
        records = []
        gesture_to_label = {g: i for i, g in enumerate(gestures)}
        
        logger.info(f"Starting training routine: {len(gestures)} gestures, {iterations} iterations each")
        
        for gesture in gestures:
            self._countdown(gesture, pause_time)
            logger.info(f"Training: {gesture}")
            
            for i in range(iterations):
                if self.verbose and i % 5 == 0 and i!=0:
                    logger.info(f"{gesture} iteration {i}/{iterations}")
                    self._countdown(gesture, pause_time-2)
                
                # Acquire measurement (extractor computes magnitudes once)
                measurement = self._acquire_measurement()
                features = measurement['features']

                # Build record with metadata
                record = {
                    'gesture': gesture,
                    'iteration': i,
                    'label': gesture_to_label[gesture],
                }
                
                # Extract S11 and S21 magnitudes from feature vector
                # Features are concatenated as [S11_0..199, S21_0..199]
                feature_array = features.x
                s11_mag = feature_array[:200]
                s21_mag = feature_array[200:400]
                
                for j, val in enumerate(s11_mag):
                    record[f'S11_{j}'] = val
                
                for j, val in enumerate(s21_mag):
                    record[f'S21_{j}'] = val

                records.append(record)
        
        # Save to CSV
        self.training_data = pd.DataFrame.from_records(records)
        output_path = Path(output_filename)
        self.training_data.to_csv(output_path, index=False)
        
        logger.info(f"Training complete: {len(records)} samples saved to {output_path}")
        
        return self.training_data

