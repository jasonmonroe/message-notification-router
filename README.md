# Message Notification Router

An AI-powered system for WhatsApp that decides which messages deserve immediate attention, which should wait, and which should be muted.

Built for the [HackerRank Orchestrate](https://www.hackerrank.com/contests/hackerrank-orchestrate-august26/challenges/message-notification-router) hackathon challenge. For the full task specification, see [`problem_statement.md`](./problem_statement.md).

## Intellectual Property & Licensing

- **AI Logic & Core Implementations**: All original agentic flows, multi-modal routing logic, system prompting structures, and classification workflows are authored by Jason Monroe of MONROE LABS CO. and licensed under the [MIT License](./LICENSE) included in this repository.
- **Starter Code & Challenge Data**: Source files, evaluation boundaries, problem scopes, and raw datasets (located within the `dataset/` and `media/` directories) are the sole property of HackerRank and are used exclusively in compliance with the [HackerRank Orchestrate Hackathon framework](https://github.com/interviewstreet/hackerrank-orchestrate-august26).

---

## Overview

WhatsApp users receive family chats, society notices, school updates, co-worker messages, business promotions, image posters, voice notes, and scams in the same stream. Treating every message the same leads to missed important updates and unwanted interruptions.

This project implements a **message notification router** that, for every incoming message, decides whether the receiving user should be:

| Action | Meaning |
|---|---|
| `notify` | Important enough to interrupt now |
| `digest` | Useful, but can be shown later |
| `mute` | Low-value, repetitive, unwanted, suspicious, or unsafe |

Decisions are **personalized** per user using message content, sender context, group/business metadata, behavioral history, and multimodal media analysis (OCR for images, Whisper transcription for voice notes).

---

## Root Directory Structure

Complete layout of all project-related files. Files marked *(generated)* or *(local only)* are not required in submission packages.

```text
message-notification-router/
│
├── main.py                              # Application entry point
├── requirements.txt                     # Python package dependencies
├── problem_statement.md                 # Full HackerRank challenge specification
├── README.md                            # This file
├── README_orig.md                       # Original starter README (reference)
├── LICENSE                              # MIT License (Monroe Labs Co)
├── AGENTS.md                            # AI agent rules and transcript logging contract
├── CLAUDE.md                            # Pointer to AGENTS.md for Claude Code
├── .env.example                         # Environment variable template (copy to .env)
├── .env                                 # Local API credentials (gitignored, do not commit)
├── .gitignore                           # Git exclusions (.env, venv, output.csv, etc.)
│
├── log.txt                              # Pipeline execution log (prompts, responses, timing) *(generated)*
├── log copy.txt                         # Archived log backup *(local only)*
├── output_orig.csv                      # Previous output run backup *(local only)*
├── event_history.csv                    # Derived event history export *(local only)*
│
├── evaluation/
│   └── main.py                          # Data pipeline and message review orchestration
│
├── models/
│   ├── __init__.py
│   └── chat_processor_model.py          # OpenAI-compatible LLM client, JSON parsing, rate-limit handling
│
├── src/
│   ├── __init__.py
│   ├── constants.py                     # System instructions, prompt templates, config constants
│   ├── data_handler.py                  # CSV loading, EDA, output persistence
│   ├── context_assembler.py             # Per-message context aggregation and media processing
│   ├── prompt_builder.py                # Structured XML prompt assembly
│   ├── utils.py                         # Logging, timers, progress bar, banner utilities
│   └── chat_transcript_logger.py        # JSON session transcript logger utility
│
├── tests/
│   └── chat_processor_model_test.py     # Unit tests for LLM response parsing and rate limits
│
└── dataset/                             # Participant-facing data (required for evaluation)
    ├── messages.csv                     # Production input — 100 messages to route
    ├── sample_messages.csv              # Labeled examples with expected output columns
    ├── output.csv                       # Submission output (written by pipeline) *(generated)*
    ├── users.csv                        # User notification behavior (quiet hours, opens, dismissals)
    ├── groups.csv                       # Group chat metadata (type, size, admins, activity)
    ├── group_members.csv                # User–group relationships (role, mute state, engagement)
    ├── business_accounts.csv            # Business sender metadata (verification, domain, reports)
    ├── user_business_history.csv        # User–business interaction history (orders, opt-outs)
    ├── message_history.csv              # Historical messages for pattern and evidence retrieval
    ├── message_events.csv               # User reactions to history (open, reply, dismiss, mute, report)
    ├── images.csv                       # Image message IDs and file paths
    ├── voice_notes.csv                  # Voice note IDs and file paths
    ├── daily_notification_summary.csv   # Daily notification load per user
    └── media/
        ├── images/
        │   ├── img_001.jpg
        │   ├── img_002.jpg
        │   ├── img_003.jpg
        │   ├── img_004.jpg
        │   ├── img_005.jpg
        │   ├── img_006.jpg
        │   ├── img_007.jpg
        │   ├── img_008.jpg
        │   ├── img_010.jpg
        │   ├── img_011.jpg
        │   ├── img_012.jpg
        │   ├── img_013.jpg
        │   ├── img_014.jpg
        │   ├── img_016.jpg
        │   ├── img_020.jpg
        │   ├── img_022.jpg
        │   ├── img_023.jpg
        │   ├── img_024.jpg
        │   ├── img_025.jpg
        │   └── img_026.jpg
        └── audio/
            ├── vn_001.mp3
            ├── vn_002.mp3
            ├── vn_003.mp3
            ├── vn_004.mp3
            ├── vn_005.mp3
            ├── vn_006.mp3
            ├── vn_007.mp3
            ├── vn_008.mp3
            ├── vn_009.mp3
            ├── vn_012.mp3
            ├── vn_013.mp3
            ├── vn_014.mp3
            └── vn_015.mp3
```

### File roles at a glance

| Path | Purpose |
|---|---|
| `main.py` | Loads `.env`, parses CLI flags, runs the full pipeline, writes `dataset/output.csv` |
| `evaluation/main.py` | `run_data_pipeline()` loads CSVs; `run_message_reviewer_pipeline()` loops messages through LLM |
| `src/context_assembler.py` | Filters user/group/business/history context; runs OCR (Tesseract) and ASR (Whisper) on media |
| `src/prompt_builder.py` | Builds XML-structured routing prompts with historical evidence and behavior profiles |
| `models/chat_processor_model.py` | Calls LLM API with `temperature=0.0`, parses JSON, handles 429 rate limits with retry |
| `src/constants.py` | `SYSTEM_INSTRUCTIONS`, `ROUTING_PROMPT_TEMPLATE`, retry timers, CSV column definitions |
| `src/data_handler.py` | Loads all dataset CSVs; `--sample` swaps in shuffled `sample_messages.csv` |
| `src/utils.py` | Appends stage logs to `log.txt`; progress bars and run timing |
| `tests/chat_processor_model_test.py` | Mocked unit tests — no live API calls required |

---

## Architecture

The pipeline processes each row in `dataset/messages.csv` sequentially:

```text
main.py
  └── evaluation/main.py
        ├── run_data_pipeline()        → DataHandler loads all CSVs
        └── run_message_reviewer_pipeline()
              ├── ContextAssembler     → Filters context per user/message
              │     ├── OCR (Tesseract) for image messages
              │     └── ASR (Whisper) for voice notes
              ├── PromptBuilder        → Builds structured XML prompt
              └── ChatProcessorModel   → Calls LLM API, parses JSON response
                    └── Writes rows to dataset/output.csv
```

### Routing logic

For each incoming message, the system:

1. **Filters context** by `user_id`, `group_id`, and `business_id` from the dataset tables.
2. **Processes media** — extracts OCR text from images (OpenCV + Tesseract) or transcribes voice notes (Whisper `base` model).
3. **Computes engagement metrics** — sender open/reply rates, notification fatigue, and recent evidence message IDs from `message_history.csv` + `message_events.csv`.
4. **Builds an XML prompt** containing user preferences, sender verification, group state (including mute status), business relationship, and the incoming message.
5. **Calls the LLM** with system instructions that enforce conflict resolution (e.g., time-sensitive logistics can override a muted group).
6. **Parses JSON output** into the required six-column submission format.

Key configuration in `src/constants.py`:

| Constant | Default | Description |
|---|---|---|
| `MAX_TOKENS` | 4096 | Max completion tokens per LLM call |
| `MAX_HISTORICAL_MESSAGES` | 5 | Historical messages included in prompt context |
| `PAUSE_TIMER` | 1.5 s | Delay between API calls to avoid rate limits |
| `RATE_LIMIT_RETRIES` | 3 | Max retry attempts on 429 errors |
| `RATE_LIMIT_PAUSE_TIMER` | 30 s | Fallback pause when retry delay cannot be parsed |

---

## Input Schema

Each row in `dataset/messages.csv` represents one incoming message:

| Field | Description |
|---|---|
| `message_id` | Unique incoming message ID |
| `user_id` | User receiving the message |
| `conversation_type` | `personal`, `group`, or `business` |
| `group_id` | Group ID (if from a group chat) |
| `business_id` | Business ID (if from a business account) |
| `sender_user_id` | Sender user ID (if from a user) |
| `created_at` | Message timestamp |
| `message_text` | Text content; empty for voice-note messages |
| `media_type` | Empty, `image`, or `voice` |
| `media_id` | Linked image or voice-note ID |
| `forwarded_count` | Forwarding signal |

### Dataset context files

| File | Used for |
|---|---|
| `users.csv` | Quiet hours, recent opens/replies/dismissals/reports |
| `groups.csv` | Group type, size, admins, recent activity |
| `group_members.csv` | User role, mute state, read/reply/dismiss behavior |
| `business_accounts.csv` | Brand identity, verification, domain, account age, reports |
| `user_business_history.csv` | Orders, bookings, payments, opt-ins/opt-outs |
| `message_history.csv` | Repeated patterns, ignored messages, useful updates, risky content |
| `message_events.csv` | Opened, replied, dismissed, muted, reported reactions |
| `images.csv` / `voice_notes.csv` | Media file path lookups |
| `daily_notification_summary.csv` | Notification load and dismissal rates per user |
| `sample_messages.csv` | Labeled examples showing expected output format and reasoning style |

---

## Requirements

- **Python 3.10+** recommended
- An **OpenAI-compatible LLM API** (endpoint, model name, and API key)
- **Tesseract OCR** installed on the system (required by `pytesseract` for image text extraction)
- **FFmpeg** (required by Whisper for audio transcription)

### Python dependencies

```bash
pip install -r requirements.txt
```

| Package | Purpose |
|---|---|
| `openai` | LLM API client |
| `pandas` | CSV data loading and filtering |
| `python-dotenv` | Environment variable management |
| `opencv-python` | Image loading and preprocessing for OCR |
| `pytesseract` | OCR text extraction from images |
| `openai-whisper` | Speech-to-text for voice notes |
| `librosa` / `tinytag` | Audio analysis and metadata (EDA mode) |
| `numpy` | Numerical operations |

---

## Setup

1. **Clone the repository** and enter the project directory.

2. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # macOS / Linux
   # .venv\Scripts\activate    # Windows
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables** — copy the example file and fill in your API credentials:

   ```bash
   cp .env.example .env
   ```

   Required variables in `.env`:

   ```text
   APP_NAME="Message Notification Router"
   MODEL_NAME=<your-model-name>
   MODEL_API_KEY=<your-api-key>
   MODEL_API_URL=<your-api-base-url>
   ```

   Never commit `.env` to version control. Secrets are read from environment variables only.

5. **Install system dependencies** (if not already present):

   ```bash
   # macOS (Homebrew)
   brew install tesseract ffmpeg

   # Ubuntu / Debian
   sudo apt install tesseract-ocr ffmpeg
   ```

---

## Running

### Full pipeline (all messages)

Process every row in `dataset/messages.csv` and write predictions to `dataset/output.csv`:

```bash
python main.py
```

### CLI flags

| Flag | Description |
|---|---|
| `--sample` | Use `dataset/sample_messages.csv` instead of `messages.csv` (rows are shuffled) |
| `--eda` | Run exploratory data analysis on loaded messages and media |
| `--log` | Enable additional logging (reserved flag) |

Examples:

```bash
# Quick test run against labeled sample data
python main.py --sample

# Explore the dataset before running the full pipeline
python main.py --eda

# Sample data with EDA
python main.py --sample --eda
```

Each run generates a unique run ID, logs prompts and LLM responses to `log.txt`, and prints progress to the console. The pipeline pauses briefly between API calls (`PAUSE_TIMER = 1.5s`) and retries on rate-limit errors.

---

## Output Format

For every row in `dataset/messages.csv`, the pipeline produces one row in `dataset/output.csv`:

```text
message_id,action,message_type,reason,confidence,evidence_message_ids
```

| Column | Description |
|---|---|
| `message_id` | Incoming message ID |
| `action` | `notify`, `digest`, or `mute` |
| `message_type` | Best-fit category (see allowed values below) |
| `reason` | Short human-readable explanation |
| `confidence` | Float from `0.0` to `1.0` |
| `evidence_message_ids` | Semicolon-separated historical message IDs, or `none` |

### Allowed values

**`action`:** `notify` · `digest` · `mute`

**`message_type`:** `personal` · `urgent` · `event` · `payment` · `business_update` · `promotion` · `greeting` · `forward` · `spam` · `scam` · `unknown`

See `dataset/sample_messages.csv` for examples of expected output style and reasoning.

Example output row:

```text
sample_msg_001,notify,urgent,A trusted group admin sent a time-sensitive update that should interrupt the user.,0.89,message_0001
```

---

## Testing

Unit tests cover credential validation, LLM response parsing, evidence ID formatting, and rate-limit retry logic (mocked — no live API calls):

```bash
# Run all tests
python3 -m unittest tests/chat_processor_model_test.py

# Run a single test class
python3 -m unittest tests.chat_processor_model_test.TestGetDelayTime

# Run a single test method
python3 -m unittest tests.chat_processor_model_test.TestGetDelayTime.test_integer_delay
```

Use `--sample` mode to validate end-to-end behavior against labeled examples before running the full dataset.

---

## Evaluation

Submissions are scored against hidden ground-truth labels. Criteria include:

- Correctness of `action`
- Correctness of `message_type`
- Usefulness and consistency of `reason`
- Relevance of `evidence_message_ids`
- Reasonable confidence calibration

The system combines structured metadata retrieval, behavioral history, OCR/ASR media processing, safety checks, and LLM contextual reasoning to make personalized routing decisions.

---

## Chat Transcript Logging

During development, compatible AI tools append conversation summaries to a shared log file (see [`AGENTS.md`](./AGENTS.md)):

| Platform | Path |
|---|---|
| macOS / Linux | `$HOME/hackerrank_orchestrate_august26/log.txt` |
| Windows | `%USERPROFILE%\hackerrank_orchestrate_august26\log.txt` |

The pipeline also writes execution logs (prompts, LLM responses, timing) to `log.txt` in the project root via `src/utils.py`.

Upload the development log as your **chat transcript** at submission time. Do not include secrets in logs.

---

## Submission

Submit the following as instructed by HackerRank:

| File | Description |
|---|---|
| `code.zip` | Full runnable solution, prompts/configs, and this README |
| `output.csv` | Predictions for all rows in `dataset/messages.csv` |
| `chat_transcript` | Development conversation log |

Before submitting, confirm:

- [ ] `output.csv` has exactly one row per `message_id` in `dataset/messages.csv`
- [ ] Columns are in the exact required order: `message_id,action,message_type,reason,confidence,evidence_message_ids`
- [ ] No secrets are hardcoded or committed
- [ ] Setup and run instructions are included in `code.zip`
- [ ] Solution reads only from `dataset/` (no organizer-only files or hardcoded labels)

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `Credentials aren't properly being read` | Ensure `.env` exists with `MODEL_NAME`, `MODEL_API_KEY`, and `MODEL_API_URL` set |
| Rate limit / 429 errors | Pipeline auto-retries with parsed delay; increase `PAUSE_TIMER` in `constants.py` if needed |
| OCR returns empty text | Confirm Tesseract is installed: `tesseract --version` |
| Whisper transcription fails | Confirm FFmpeg is installed: `ffmpeg -version` |
| Pipeline stops mid-run | Check `log.txt` for the last successful `LLM_RESPONSE`; empty responses halt the loop |
| Missing `output.csv` rows | Re-run from scratch; `save_output()` upserts by `message_id` |

---

## Author

Jason Monroe — [jason@jasonmonroe.com](mailto:jason@jasonmonroe.com)  
Copyright © 2011–2026 Monroe Labs Co
