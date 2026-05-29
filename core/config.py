# RMGrealtime/core/config.py
from pathlib import Path

#LiteVNA port
#For Laptop
COM_PORT = "COM3"
#For RPI5
# COM_PORT = "/dev/ttyACM0"

START_FREQ = 2.5e9
STOP_FREQ = 3.5e9
POINTS = 100

BULK_READ = 20

# Path to this file: .../RMGrealtime/core/config.py
_BASE_DIR = Path(__file__).resolve().parent.parent  # goes up to RMGrealtime/

# Define a shared calibration file path
CAL_FILE = _BASE_DIR / "rf" / "LiteVNA.cal"

#Linker hand
CAN_INTERFACE = "can0"
HAND_TYPE = "right"
HAND_JOINT = "L7"
