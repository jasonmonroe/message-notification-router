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
MAX_HISTORICAL_MESSAGES = 5
MSEC = 1000
SECS_IN_MIN = 60
PAUSE_TIMER = 1.5
RATE_LIMIT_PAUSE_TIMER = 30
RATE_LIMIT_RETRIES = 3
PEP8_LINE_LEN = 79

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
SYSTEM_INSTRUCTIONS = """
You are a machine learning expert with extensive knowledge in multimodal prompts for an AI-powered system such as WhatsApp that decides which messages deserve immediate attention, which should wait, and which should be muted.

## CRITICAL EXECUTION RULES:
1. Analyze xml data that consists of messages, user context, business context, group metadata, history and media attachments (if available) together to make a routing determination.
2. You must output your final routing determination strictly as a valid JSON object matching the requested schema, using exact action values: 'notify', 'digest', or 'mute'.
---

## CONFIDENCE SCORING CRITERIA:
Evaluate your certainty for the chosen action on a scale from 0.0 to 1.0:
- 1.0: Absolute certainty, clear intent, rich context matching user preferences.
- 0.8 - 0.9: High confidence, minor ambiguity, or standard routing logic applied.
- 0.5 - 0.7: Moderate uncertainty, mixed signals, or slight guesswork.
- Below 0.5: High ambiguity, highly unpredictable content, or missing critical context.
---

## CONFLICT RESOLUTION POLICY:
- Active, time-sensitive logistical coordination (such as an imminent meetup, payment, or pickup scheduled for today/this weekend) SHOULD override a group's 'muted' status if historical evidence shows an ongoing, active thread between parties.
- General chat, promotional broadcasts, or automated alerts from a muted group must remain 'muted' or sent to 'digest'.
""".strip()

ROUTING_PROMPT_TEMPLATE = """
## USER - BUSINESS MESSAGE DATA

<context>
    {incoming_message_context}

    {business_sender_context}

    {recipient_user_context}

    {group_metadata_context}

    {historical_evidence}

    {media_context}
</context>


## TASK INSTRUCTION
Analyze the incoming message, user preferences, sender verification, and history. 
Pay special attention to whether an incoming message represents an active, time-sensitive transaction that overrides a muted group state.
Decide whether this message should be:
1. `notify` (interrupt now)
2. `digest` (save for later)
3. `mute` (suppress as low-value, repetitive, or unsafe)
 
CRITICAL OUTPUT REQUIREMENT:
Return your response as a valid JSON object wrapped inside a markdown code block (```json ... ```). 

**The JSON structure below is a template/blueprint.** Do not use the sample IDs or values from it. Populate all keys using the *actual data, IDs, and decisions* derived from the prompt context above:

{{
    "message_id": "sample_msg_041",
    "action": "notify", 
    "message_type": "transaction",
    "reason": "Explain your decision here...",
    "confidence": 0.85,
    "evidence_message_ids": ["message_0046"] 
}}

**IMPORTANT RULES:**
1. JSON object keys must be in the exact order shown above. Do not deviate!
2. Replace all placeholder values with real data from the current context.
""".strip()
