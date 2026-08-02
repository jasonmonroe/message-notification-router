# code/src/constants.py

import os

# +-----------------------------------+
# |           CONSTANTS               |
# +-----------------------------------+

ARGS_LIST = [
    "--build",
    "--log",
    "--sample",
]

# Misc 
MAX_TOKENS = 4096
TOKEN_UNIT = 1000
SECS_IN_MIN = 60
PAUSE_TIMER = 1
SLEEP_TIMER = 20
SLEEP_TIMER_INC = 4
RISK_FLAG_CNT = 5
MSEC = 1000
PEP8_LINE_LEN = 79

# Dataset files
DATASET_DIR = "../dataset/"
AUDIO_DIR = f"{DATASET_DIR}media/audio/"
IMAGES_DIR = f"{DATASET_DIR}media/images/"

AUDIO_SAMPLE_RATE = None

CHAT_TRANSCRIPT_FILE = "log.txt"
OUTPUT_FILE = "output.csv"

# Model Information
APP_NAME = os.getenv("APP_NAME")
MODEL_API_URL = os.getenv("MODEL_API_URL")
MODEL_API_KEY = os.getenv("MODEL_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
