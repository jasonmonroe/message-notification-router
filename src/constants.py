# code/src/constants.py

import os

# +-----------------------------------+
# |           CONSTANTS               |
# +-----------------------------------+

ARGS_LIST = [
    "--eda",
    "--log",
    "--sample",
]

# Misc 
MAX_TOKENS = 4096
SECS_IN_MIN = 60
PAUSE_TIMER = 1
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

CSV_HEADER_COLS = [
    "message_id", 
    "action", 
    "message_type", 
    "reason",
    "confidence", 
    "evidence_message_ids", 
]

# Dataset files
CHAT_TRANSCRIPT_FILE = "log.txt"
DATASET_DIR = "dataset/"
OUTPUT_FILEPATH = os.path.join(DATASET_DIR, "output.csv")

AUDIO_DIR = os.path.join(DATASET_DIR, "media", "audio")
IMAGES_DIR = os.path.join(DATASET_DIR, "media", "images")
AUDIO_SAMPLE_RATE = None

# Model Information
APP_NAME = os.getenv("APP_NAME")
MODEL_API_KEY = os.getenv("MODEL_API_KEY")
MODEL_API_URL = os.getenv("MODEL_API_URL")
MODEL_NAME = os.getenv("MODEL_NAME")

# Prompt Information
SYSTEM_INSTRUCTIONS = """You are a machine learning expert with extensive knowledge in multimodal prompts for an AI-powered system such as WhatsApp that decides which messages deserve immediate attention, which should wait, and which should be muted.

## CRITICAL EXECUTION RULES:
1. Analyze text data, user context, group metadata, and history together to make a routing determination.
2. You must output your final routing determination strictly as a valid JSON object matching the requested schema, using exact action values: 'notify', 'digest', or 'mute'.
-----------------------------------------------

## CONFIDENCE SCORING CRITERIA:
Evaluate your certainty for the chosen action on a scale from 0.0 to 1.0:
- 1.0: Absolute certainty, clear intent, rich context matching user preferences.
- 0.8 - 0.9: High confidence, minor ambiguity, or standard routing logic applied.
- 0.5 - 0.7: Moderate uncertainty, mixed signals, or slight guesswork.
- Below 0.5: High ambiguity, highly unpredictable content, or missing critical context.
-----------------------------------------------

## CONFLICT RESOLUTION POLICY:
- Active, time-sensitive logistical coordination (such as an imminent meetup, payment, or pickup scheduled for today/this weekend) SHOULD override a group's 'muted' status if historical evidence shows an ongoing, active thread between parties.
- General chat, promotional broadcasts, or automated alerts from a muted group must remain 'muted' or sent to 'digest'.
""".strip()


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
Analyze the incoming message, user preferences, sender verification, and history. 
Pay special attention to whether an incoming message represents an active, time-sensitive transaction that overrides a muted group state.
Decide whether this message should be:
1. `notify` (interrupt now)
2. `digest` (save for later)
3. `mute` (suppress as low-value, repetitive, or unsafe)
-----------------------------------------------

CRITICAL OUTPUT REQUIREMENT:
Return your response as a valid JSON object wrapped inside a markdown code block (```json ... ```) matching this exact schema:
{{
    "message_id": "{message_id}",
    "action": "notify", 
    "message_type": "transaction",
    "reason": "Explain your decision here...",
    "confidence": 0.85,
    "evidence_message_ids": ["message_0232", "message_0335"]
}}

** IMPORTANT: The object keys need to be in this order.  Do not deviate! **
""".strip()
