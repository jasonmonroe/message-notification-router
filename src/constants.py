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
MAX_HISTORICAL_MESSAGES = 5

# Dataset file names
CSV_FILENAMES = [
    "business_accounts",
    "daily_notification_summary",
    "group_members",
    "groups",
    "images",
    "message_events",
    "message_history",
    "messages",
    "output",
    "user_business_history",
    "users",
    "voice_notes"
]

CSV_OUTPUT_COLS = [
    "message_id", 
    "action", 
    "message_type", 
    "reason",
    "confidence", 
    "evidence_message_ids", 
]

# Dataset files
DATASET_DIR = "dataset/"
AUDIO_DIR = os.path.join(DATASET_DIR, "media", "audio")
IMAGES_DIR = os.path.join(DATASET_DIR, "media", "images")
CHAT_TRANSCRIPT_FILE = "log.txt"
OUTPUT_FILE = "output.csv"

AUDIO_SAMPLE_RATE = None

# Model Information
APP_NAME = os.getenv("APP_NAME")
MODEL_API_KEY = os.getenv("MODEL_API_KEY")
MODEL_API_URL = os.getenv("MODEL_API_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

# Prompt Information
SYSTEM_INSTRUCTIONS = (
    "You are a machine learning expert with extensive knowledge in multimodal prompts for an AI-powered system "
    "such as WhatsApp that decides which messages deserve immediate attention, which should wait, and which should be muted.\n\n"
    "CRITICAL EXECUTION RULES:\n"
    "1. Analyze text data, user context, group metadata, and history together to make a routing determination.\n"
    "2. You must output your final routing determination strictly as a valid JSON object matching the requested schema, "
    "using exact action values: 'notify', 'digest', or 'mute'."
).strip()

ROUTING_PROMPT_TEMPLATE = """
## INCOMING MESSAGE TO ROUTE
{incoming_message_context}
-----------------------------------------------

## MEDIA ATTACHMENT CONTEXT (Type: {media_type}, File: {media_filename})
{media_content_description}
-----------------------------------------------

## RECIPIENT USER CONTEXT (User ID: {user_id})
{recipient_user_context}
-----------------------------------------------

## BUSINESS SENDER CONTEXT (Business ID: {business_id})
{business_sender_context}
-----------------------------------------------

## RECENT HISTORICAL EVIDENCE (Optional)
{historical_evidence}
-----------------------------------------------

## GROUP METADATA CONTEXT (Group ID: {group_id})
{group_metadata_context}
-----------------------------------------------

## TASK INSTRUCTION
Analyze the incoming message, user preferences, sender verification, and history. Decide whether this message should be:
1. `notify` (interrupt now)
2. `digest` (save for later)
3. `mute` (suppress as low-value, repetitive, or unsafe)

CRITICAL OUTPUT REQUIREMENT:
Return your response as a valid JSON object wrapped inside a markdown code block (```json ... ```) matching this exact schema:
{{
  "message_id": "{message_id}",
  "action": "...",
  "message_type": "...",
  "reason": "...",
  "confidence": 0.95,
  "evidence_message_ids": ["..."]
}}
""".strip()
