# code/src/constants.py

import os

# +-----------------------------------+
# |           CONSTANTS               |
# +-----------------------------------+

ARGS_LIST = [
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
CSV_FILENAMES = [
    "business_accounts",
    "daily_notification_summary",
    "group_members",
    "groups",
    "images",
    "message_events",
    "message_history",
    "messages",
    #"output",
    "user_business_history",
    "users",
    "voice_notes"
]

DATASET_DIR = "dataset/"
AUDIO_DIR = os.path.join(DATASET_DIR, "media", "audio")
IMAGES_DIR = os.path.join(DATASET_DIR, "media", "images")

AUDIO_SAMPLE_RATE = None

CHAT_TRANSCRIPT_FILE = "log.txt"
OUTPUT_FILE = "output.csv"

# Model Information
APP_NAME = os.getenv("APP_NAME")
MODEL_API_KEY = os.getenv("MODEL_API_KEY")
MODEL_API_URL = os.getenv("MODEL_API_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
