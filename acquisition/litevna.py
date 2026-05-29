import serial
import struct
import numpy as np
import time
import logging
from typing import Generator

from core.types import Sweep

logger = logging.getLogger(__name__)

class LiteVNA:
    def __init__(self, port="COM3", start_freq = 2.5e9, stop_freq = 3.5e9, 
                 points = 200, read_size = 20, mode = 0, verbose=False):
        self.verbose = verbose
        
        if self.verbose:
            logger.info(f"Initializing LiteVNA on {port}: {start_freq/1e9:.2f}-{stop_freq/1e9:.2f} GHz, {points} points")
        
        self.ser = serial.Serial(port, 115200, timeout=1)

        self.start_freq = start_freq
        self.stop_freq = stop_freq
        self.points = points
        self.step_freq = (self.stop_freq - self.start_freq) / (self.points - 1)
        self.freqs = np.linspace(self.start_freq, self.stop_freq, self.points)

        self.read_size = read_size

        self.fwd = np.zeros(points, dtype=np.complex64)
        self.rev0 = np.zeros(points, dtype=np.complex64)
        self.rev1 = np.zeros(points, dtype=np.complex64)
        self.valid = np.zeros(points, dtype=bool)

        self.last_idx = -1
        
        self.channel_mode = mode
        self.configure(channel_mode=self.channel_mode)

    # ============================================================
    # LOW LEVEL WRITE HELPERS (RESTORED)
    # ============================================================

    def _write1(self, addr, value):
        self.ser.write(bytes([0x20, addr, value]))

    def _write2(self, addr, value):
        self.ser.write(bytes([0x21, addr]) + struct.pack('<H', value))

    def _write8(self, addr, value):
        self.ser.write(bytes([0x23, addr]) + struct.pack('<Q', int(value)))

    # ============================================================
    # CONFIGURATION (NEW CLEAN ENTRY POINT)
    # ============================================================

    def configure(self, channel_mode=0x00):
        """
        Configures LiteVNA hardware.
        Must be called once before streaming/acquisition.
        """
        
        if self.verbose:
            logger.info("Configuring LiteVNA hardware")

        # frequency setup
        self._write8(0x00, int(self.start_freq))
        self._write8(0x10, int(self.step_freq))
        self._write2(0x20, self.points)
        self._write2(0x22, 1)

        # channel mode:
        # 0x00 = S11 + S21
        # 0x01 = S11 only
        # 0x02 = S21 only
        self._write1(0x44, channel_mode)

        # fast mode / streaming enable
        self._write1(0x40, 1)

    # ============================================================
    # LOW LEVEL READ
    # ============================================================

    def _read_fifo(self):
        self.ser.write(bytes([0x18, 0x30, self.read_size]))
        data = self.ser.read(32 * self.read_size)
        return data if len(data) == 32 * self.read_size else None

    # ============================================================
    # PROCESS CHUNK
    # ============================================================

    def _process(self, data):
        completed = False

        for i in range(self.read_size):

            vals = struct.unpack('<iiiiiiH6x', data[i * 32:(i + 1) * 32])

            fwd = complex(vals[0], vals[1])
            rev0 = complex(vals[2], vals[3])
            rev1 = complex(vals[4], vals[5])
            idx = vals[6]

            if idx >= self.points:
                continue

            if self.last_idx != -1 and idx < self.last_idx:
                if np.all(self.valid):
                    completed = True
                self.valid[:] = False

            self.last_idx = idx

            self.fwd[idx] = fwd
            self.rev0[idx] = rev0
            self.rev1[idx] = rev1
            self.valid[idx] = True

        return completed

    # ============================================================
    # 1. BLOCKING SWEEP (RMGrealtime acquisition contract)
    # ============================================================

    def acquire_sweep(self) -> Sweep:
        # RESET ACQUISITION STATE
        if self.verbose:
            logger.info("Acquiring sweep")
        
        self.valid[:] = False
        self.last_idx = -1
        
        while True:
            data = self._read_fifo()
            if data is None:
                continue

            if self._process(data):
                return Sweep(
                    freq=self.freqs,
                    fwd=self.fwd.copy(),
                    rev0=self.rev0.copy(),
                    rev1=self.rev1.copy()
                )

    # ============================================================
    # 2. STREAMING MODE (REAL-TIME PIPELINE)
    # ============================================================

    def stream_sweeps(self, hz_limit: float = None) -> Generator[Sweep, None, None]:
        """
        Continuous sweep generator.

        This is for:
        - real-time ML inference
        - decision layer pipelines
        - live visualization
        """
        
        if self.verbose:
            logger.info(f"Starting sweep stream with hz_limit={hz_limit}")

        last_time = time.perf_counter()

        while True:
            data = self._read_fifo()
            if data is None:
                continue

            completed = self._process(data)

            if not completed:
                continue

            now = time.perf_counter()

            if hz_limit is not None:
                dt = now - last_time
                min_dt = 1.0 / hz_limit
                if dt < min_dt:
                    continue

            last_time = now

            yield Sweep(
                freq=self.freqs,
                fwd=self.fwd.copy(),
                rev0=self.rev0.copy(),
                rev1=self.rev1.copy()
            )