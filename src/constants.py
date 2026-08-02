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
    #"output",
    "user_business_history",
    "users",
    "voice_notes"
]

# Dataset files
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


# Prompt Information
SYSTEM_INSTRUCTIONS = (
    "You are a machine learning expert with extensive knowledge in multimodal prompts for an AI-powered system "
    "for WhatsApp that decides which messages deserve immediate attention, which should wait, and which should be muted.\n\n"
    "CRITICAL EXECUTION RULES:\n"
    "1. Analyze text data, audio duration, and image context together to make a routing determination.\n"
    "2. If an audio file is corrupt or missing (duration/shape is zero), default the message to 'wait'.\n"
    "3. You must output your final routing determination strictly as a JSON object matching this schema:\n"
    "   {\"routing_decision\": \"immediate\" | \"wait\" | \"muted\", \"reason\": \"string reason\"}"
).strip()


SYSTEM_PROMPT = """
You are an automated message router tasked with prioritizing message notificications.  Analyze the user details, text metadata, and incoming media files to generate routing decisions.


""".strip()


ROUTING_PROMPT_TEMPLATE = """
## INCOMING MESSAGE TO ROUTE
{incoming_message_context}
-----------------------------------------------

## RECIPIENT USER CONTEXT (User ID: {user_id})
{recipient_user_context}
---

## BUSINESS SENDER CONTEXT (Business ID: {business_id})
{business_sender_context}
---

## RECENT HISTORICAL EVIDENCE (Optional)
{historical_evidence}

## TASK INSTRUCTION
Analyze the incoming message, user preferences, sender verification, and history. Decide whether this message should be:
1. `notify` (interrupt now)
2. `digest` (save for later)
3. `mute` (suppress as low-value, repetitive, or unsafe)

Return your response as a valid JSON object wrapped inside a markdown code block (```json ... ```) matching this exact schema:
{{
  "message_id": "{message_id}",
  "action": "...",
  "message_type": "...",
  "reason": "...",
  "confidence": 0.95,
  "evidence_message_ids": "none"
}}
""".strip()



# Example
MESSAGE_CONTENT_PROMPT = """
## INCOMING MESSAGE TO ROUTE
{}
- Message ID: msg_023
- Conversation Type: business
- Sender / Business ID: business_002
- Target User ID: u_023
- Created At: 2026-07-30 22:19
- Message Text: Important Information\n\nDear Customer,\n\nYour latest account or card payment update is now available.\n\nPlease review the details and next steps in your banking app.\n\nTeam Banking Services
- Forwarded Count: 0

---

## RECIPIENT USER CONTEXT (User ID: u_023)
- Do Not Disturb Window: 21:30-07:30
- 30-Day Activity: 74 opened, 18 replied, 31 dismissed, 1 reported

---

## BUSINESS SENDER CONTEXT (Business ID: business_002)
- Brand Name: [Look up from business_accounts.csv]
- Verified Status: [Look up from business_accounts.csv]
- User Business History: [Look up recent interactions from user_business_history.csv for u_023 & business_002]

---

## RECENT HISTORICAL EVIDENCE (Optional)
- Past similar messages received by this user and how they reacted (from message_history.csv & message_events.csv).

---

## TASK INSTRUCTION
Analyze the incoming message, user preferences, sender verification, and history. Decide whether this message should be:
1. `notify` (interrupt now)
2. `digest` (save for later)
3. `mute` (suppress as low-value, repetitive, or unsafe)

Return your response in strict JSON matching the required output schema:
{
  "message_id": "msg_023",
  "action": "...",
  "message_type": "...",
  "reason": "...",
  "confidence": 0.95,
  "evidence_message_ids": "none"
}

""".strip()