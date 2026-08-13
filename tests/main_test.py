# tests/main_test.py

# +-----------------------------------------------------------------------------+
# |                                 UNIT TESTS                                  |
# +-----------------------------------------------------------------------------+
# | Unit Testing main.py                                                        |
# +-----------------------------------------------------------------------------+

# Python Libraries
import os
import sys
import unittest
from unittest.mock import call, patch, MagicMock

# Vendor Libraries
import pandas as pd

from src.constants import ARGS_LIST, CSV_FILENAMES

# Dynamically add the project root directory (one level up from 'tests') to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local Libraries
from evaluation.main import run_data_pipeline, run_message_reviewer_pipeline
import main
from models import chat_processor_model
from src.data_handler import DataHandler


@patch('evaluation.main.log_chat_transcript')
class TestMainPipeline(unittest.TestCase):

    def setUp(self):
        """Runs before every test."""
        self.mock_client = MagicMock()

    def test_it_parses_args(self, mock_log_transcript):
        expected = {"eda": True, "log": True, "sample": True}
        actual = main.parse_args(ARGS_LIST)
        print(actual)
        self.assertIsInstance(expected, dict)
        self.assertEqual(actual, expected)
        self.assertHasAttr(actual, "eda")
        self.assertHasAttr(actual, "log")
        self.assertHasAttr(actual, "sample")

    def test_it_returns_data_handler(self, mock_log_transcript):
        args = {"sample": True, "eda": True}

        # Run the pipeline function under test
        actual = run_data_pipeline(args)

        # Verify returned object type
        self.assertIsInstance(actual, DataHandler)

        # Verify object attributes on the returned instance
        expected_attrs = [
            "business_accounts", "daily_notification_summary", "group_members",
            "groups", "images", "message_events", "message_history",
            "messages", "output", "user_business_history", "users", "voice_notes"
        ]


        for attr in CSV_FILENAMES:
            self.assertTrue(hasattr(actual, attr), f"DataHandler missing attribute: {attr}")

    @patch("main.APP_NAME", "Message Notification Router")
    @patch('models.chat_processor_model.OpenAI')
    @patch('models.chat_processor_model.time.sleep')
    def test_it_runs_message_reviewer_pipeline(
        self, mock_sleep, mock_openai_class, mock_log_transcript
    ):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the chat.completions.create() return payload
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="""{
                "action": "notify",
                "message_type": "urgent",
                "reason": "A trusted group admin sent a time-sensitive update that should interrupt the user.",
                "confidence": 0.89,
                "evidence_message_ids": ["message_0001"]
            }"""))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # Only run one row...
        row_df = pd.DataFrame([{
            "message_id": "sample_msg_001",
            "user_id": "u_011",
            "conversation_type": "group",
            "group_id": "group_002",
            "business_id": None,
            "sender_user_id": "u_043",
            "created_at": "2026-07-31 11:09",
            "message_text": (
                "Tower B folks, quick heads-up. The tanker guy is saying he can wait maybe 20 mins max because "
                "he has another stop after this. Motor room valve is still open, so if your flat missed morning supply, "
                "pls fill drinking water now. I know this is annoying, but better to store a little. Will update after 6 once plumber confirms."
            ),
            "media_type": None,
            "media_id": None,
            "forwarded_count": 0,
            "action": "notify",
            "message_type": "urgent",
            "reason": "A trusted group admin sent a time-sensitive update that should interrupt the user.",
            "confidence": 0.89,
            "evidence_message_ids": "message_0001"
        }])

        args = {"sample": True, "eda": False}
        handler = DataHandler(args)
        handler.messages = row_df

        # Execute pipeline
        result = run_message_reviewer_pipeline(handler)

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        mock_client.chat.completions.create.assert_called_once()

    @patch("main.APP_NAME", "Message Notification Router")
    @patch('models.chat_processor_model.OpenAI')
    @patch('models.chat_processor_model.time.sleep')
    def test_it_runs_message_reviewer_pipeline_with_image(
        self, mock_sleep, mock_openai_class, mock_log_transcript
    ):
        # 1. Setup client mock & JSON response payload
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="""{
                "action": "digest",
                "message_type": "promotion",
                "reason": "The message matches the user's known interests but is still low priority.",
                "confidence": 0.84,
                "evidence_message_ids": ["message_0049"]
            }"""))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # 2. Mock 1-row image payload
        row_df = pd.DataFrame([{
            "message_id": "sample_msg_044",
            "user_id": "u_032",
            "conversation_type": "group",
            "group_id": "group_005",
            "business_id": None,
            "sender_user_id": "u_048",
            "created_at": "2026-07-31 08:28",
            "message_text": "Photos for the kurta set are attached. Pickup is near Gate 2 this weekend.",
            "media_type": "image",
            "media_id": "img_008",
            "forwarded_count": 0,
            "action": "digest",
            "message_type": "promotion",
            "reason": "The message matches the user's known interests but is still low priority.",
            "confidence": 0.84,
            "evidence_message_ids": "message_0049"
        }])

        args = {"sample": True, "eda": False}
        handler = DataHandler(args)
        handler.messages = row_df

        # 3. Execute pipeline
        result = run_message_reviewer_pipeline(handler)

        # 4. Assertions
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        mock_client.chat.completions.create.assert_called_once()

    @patch("main.APP_NAME", "Message Notification Router")
    @patch('models.chat_processor_model.OpenAI')
    @patch('models.chat_processor_model.time.sleep')
    def test_it_runs_message_reviewer_pipeline_with_voice_note(
        self, mock_sleep, mock_openai_class, mock_log_transcript
    ):
        # 1. Setup client mock & JSON response payload
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="""{
                "action": "digest",
                "message_type": "personal",
                "reason": "The sender is trusted, but the message has no urgent action or safety relevance.",
                "confidence": 0.82,
                "evidence_message_ids": ["message_0046"]
            }"""))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        # 2. Mock 1-row voice note payload (message_text is None)
        row_df = pd.DataFrame([{
            "message_id": "sample_msg_041",
            "user_id": "u_024",
            "conversation_type": "group",
            "group_id": "group_008",
            "business_id": None,
            "sender_user_id": "u_041",
            "created_at": "2026-07-31 11:09",
            "message_text": None,
            "media_type": "voice",
            "media_id": "vn_001",
            "forwarded_count": 0,
            "action": "digest",
            "message_type": "personal",
            "reason": "The sender is trusted, but the message has no urgent action or safety relevance.",
            "confidence": 0.82,
            "evidence_message_ids": "message_0046"
        }])

        args = {"sample": True, "eda": False}
        handler = DataHandler(args)
        handler.messages = row_df

        # 3. Execute pipeline
        result = run_message_reviewer_pipeline(handler)

        # 4. Assertions
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        mock_client.chat.completions.create.assert_called_once()

    @patch("main.APP_NAME", "Message Notification Router")
    @patch('models.chat_processor_model.OpenAI')
    @patch('models.chat_processor_model.time.sleep')
    def test_it_runs_message_reviewer_pipeline_for_all_messages(
        self, mock_sleep, mock_openai_class, mock_log_transcript
    ):
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        # Mock the chat.completions.create() return payload
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="""{
                "action": "notify",
                "message_type": "urgent",
                "reason": "A trusted group admin sent a time-sensitive update that should interrupt the user.",
                "confidence": 0.89,
                "evidence_message_ids": ["message_0001"]
            }"""))
        ]
        mock_client.chat.completions.create.return_value = mock_response

        args = {"sample": True, "eda": False}
        handler = DataHandler(args)

        # Execute pipeline across ALL loaded messages
        result = run_message_reviewer_pipeline(handler)
        total_rows = len(handler.messages)

        # Assertions
        self.assertIsNotNone(result)
        self.assertEqual(len(result), total_rows)
        self.assertEqual(mock_client.chat.completions.create.call_count, total_rows)


if __name__ == "__main__":
    unittest.main()