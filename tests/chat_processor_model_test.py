# tests/chat_processor_model_test.py

# +-----------------------------------------------------------------------------+
# |                                 UNIT TESTS                                  |
# +-----------------------------------------------------------------------------+
# Unit Testing Chat Processor Model
#
# - Run all tests: python3 -m unittest tests/chat_processor_model_test.py
# - Run Single Test: python3 -m unittest tests.chat_processor_model_test.TestGetDelayTime.test_integer_delay
# - Run Single Class: python3 -m unittest tests.chat_processor_model_test.TestGetDelayTime

# Python Libraries
import json
import os
import random
import sys
import unittest
from unittest.mock import call, patch, MagicMock

# Vendor Libraries
from openai import RateLimitError
from openai.types.chat import ChatCompletion

# Local Libraries
from models.chat_processor_model import ChatProcessorModel
from src.constants import RATE_LIMIT_PAUSE_TIMER

import src.constants
# Set dummy environment variables *before* importing or initializing the model
src.constants.MODEL_API_KEY = "mock-api-key-for-testing"
src.constants.MODEL_API_URL = "https://site.url/api/unit-test/"
src.constants.MODEL_NAME = "gpt-4o-mini"

# Dynamically add the project root directory (one level up from 'tests') to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestChatProcessorModelAPI(unittest.TestCase):

    @patch('models.chat_processor_model.OpenAI')
    def setUp(self, mock_openai_class):
        """
        Runs before every test. We patch 'OpenAI' so __init__ 
        instantiates a fake client instead of hitting the internet.
        """
        # Create a mock instance that the OpenAI class constructor will return
        self.mock_client = MagicMock()
        mock_openai_class.return_value = self.mock_client
        
        # Instantiate your model under test (fixed randomint -> randint)
        self.row_cnt = random.randint(20, 40)
        self.model = ChatProcessorModel(row_cnt=self.row_cnt)

    @patch('models.chat_processor_model.OpenAI')
    @patch("models.chat_processor_model.MODEL_NAME", None)
    def test_it_has_improper_credentials(self, mock_openai):
        """Should raise ValueError when credential constants are None upon initialization."""
        with self.assertRaises(ValueError) as context:
            ChatProcessorModel(row_cnt=self.row_cnt)
        
        self.assertIn("🚨 Credentials aren't properly being read. Check environment file. 🚨", str(context.exception))


    @patch('models.chat_processor_model.time.sleep')
    def test_get_response_success_with_evidence_ids(self, mock_sleep):
        """Should convert populated evidence list to semicolon-separated string."""
        mock_data = {
            "message_id": "msg_12345",
            "action": "notify",
            "message_type": "transaction",
            "reason": "Explain your decision here...",
            "confidence": 0.85,
            "evidence_message_ids": ["message_0046"]
        }
        
        # Expected output after _format_response processes the list into a CSV string
        expected = {
            "message_id": "msg_12345",
            "action": "notify",
            "message_type": "transaction",
            "reason": "Explain your decision here...",
            "confidence": 0.85,
            "evidence_message_ids": "message_0046"
        }

        mock_response = MagicMock(spec=ChatCompletion)
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(mock_data)
        mock_response.choices = [mock_choice]
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        actual = self.model.get_response(prompt="Analyze this text", row_index=self.row_cnt)
     
        self.assertEqual(actual, expected)
        self.mock_client.chat.completions.create.assert_called_once()

    @patch('models.chat_processor_model.time.sleep')
    def test_get_response_success_empty_evidence_ids(self, mock_sleep):
        """Should format empty evidence list to 'none'."""
        mock_data = {
            "message_id": "msg_12345",
            "action": "notify",
            "message_type": "transaction",
            "reason": "Explain your decision here...",
            "confidence": 0.85,
            "evidence_message_ids": []
        }
        
        # Expected output after _format_response processes []
        expected = {
            "message_id": "msg_12345",
            "action": "notify",
            "message_type": "transaction",
            "reason": "Explain your decision here...",
            "confidence": 0.85,
            "evidence_message_ids": "none"
        }

        mock_response = MagicMock(spec=ChatCompletion)
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(mock_data)
        mock_response.choices = [mock_choice]
        
        self.mock_client.chat.completions.create.return_value = mock_response
        
        actual = self.model.get_response(prompt="Analyze this text", row_index=self.row_cnt)
     
        self.assertEqual(actual, expected)
        self.mock_client.chat.completions.create.assert_called_once()

    @patch('models.chat_processor_model.time.sleep')  # Prevent tests from freezing during sleeps
    def test_get_response_success(self, mock_sleep):
        """Should return formatted list values on a successful JSON API response."""
       
        expected = {
            "message_id": "msg_12345",  # Provide a realistic test string
            "action": "notify",
            "message_type": "transaction",
            "reason": "Explain your decision here...",
            "confidence": 0.85,
            "evidence_message_ids": "message_0046"
        }

        # 1. Build a structure mimicking a real OpenAI response object
        mock_response = MagicMock(spec=ChatCompletion)
        mock_choice = MagicMock()

        # Your model uses response_format={"type": "json_object"}
        mock_data = {
            "message_id": "msg_12345",  # Provide a realistic test string
            "action": "notify",
            "message_type": "transaction",
            "reason": "Explain your decision here...",
            "confidence": 0.85,
            "evidence_message_ids": ["message_0046"] 
        }

        mock_choice.message.content = json.dumps(mock_data)
       
        mock_response.choices = [mock_choice]
        
        # Tell the mock client to return this exact response structure
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Call your protected/public workflow
        actual = self.model.get_response(prompt="Analyze this text", row_index=random.randint(0, self.row_cnt-1))
     
        # Assertions: Verify data processing and underlying API calls
        self.assertEqual(actual, expected)  # _format_response extracts dict values
        self.mock_client.chat.completions.create.assert_called_once()


    @patch('models.chat_processor_model.time.sleep')  # Prevent tests from freezing during sleeps
    def test_it_returns_none_for_evidence_message_ids(self, mock_sleep):
        """Should return formatted list values on a successful JSON API response."""
       
        expected = {
            "message_id": "msg_12345",  # Provide a realistic test string
            "action": "notify",
            "message_type": "transaction",
            "reason": "Explain your decision here...",
            "confidence": 0.85,
            "evidence_message_ids": "none"
        }

        # 1. Build a structure mimicking a real OpenAI response object
        mock_response = MagicMock(spec=ChatCompletion)
        mock_choice = MagicMock()

        # Your model uses response_format={"type": "json_object"}
        mock_data = {
            "message_id": "msg_12345",  # Provide a realistic test string
            "action": "notify",
            "message_type": "transaction",
            "reason": "Explain your decision here...",
            "confidence": 0.85,
            "evidence_message_ids": None
        }

        mock_choice.message.content = json.dumps(mock_data)
       
        mock_response.choices = [mock_choice]
        
        # Tell the mock client to return this exact response structure
        self.mock_client.chat.completions.create.return_value = mock_response
        
        # Call your protected/public workflow
        actual = self.model.get_response(prompt="Analyze this text", row_index=random.randint(0, self.row_cnt-1))
     
        # Assertions: Verify data processing and underlying API calls
        self.assertEqual(actual, expected)  # _format_response extracts dict values
        self.mock_client.chat.completions.create.assert_called_once()


    @patch('models.chat_processor_model.time.sleep')
    def test_format_response_string_evidence_ids(self, mock_sleep):
        """Should pass through clean string evidence IDs or convert null string markers to 'none'."""
        
        # 1. Test populated string pass-through
        mock_data_valid = {
            "message_id": "msg_12345",
            "evidence_message_ids": "  message_0046  "  # Leading/trailing whitespace
        }
        
        mock_response = MagicMock(spec=ChatCompletion)
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(mock_data_valid)
        mock_response.choices = [mock_choice]
        self.mock_client.chat.completions.create.return_value = mock_response
        
        actual = self.model.get_response(prompt="Test", row_index=self.row_cnt)
        self.assertEqual(actual["evidence_message_ids"], "message_0046")

        # 2. Test string null marker conversion (e.g. "[]" or "none")
        mock_data_null_str = {
            "message_id": "msg_12345",
            "evidence_message_ids": "[]"
        }
        mock_choice.message.content = json.dumps(mock_data_null_str)
        
        actual_null = self.model.get_response(prompt="Test", row_index=self.row_cnt)
        self.assertEqual(actual_null["evidence_message_ids"], "none")
    

    @patch('models.chat_processor_model.time.sleep')
    def test_get_response_rate_limit_retry_flow(self, mock_sleep):
        """Should capture RateLimitError, extract delay, and trigger sleep."""
        
        expected_result = {"error": True}

        # Construct a fake raw HTTP response structure required by OpenAI errors
        mock_http_response = MagicMock()
        mock_http_response.headers = {}
        
        # Pick a random delay time to make the test more dynamic.
        delay_time = round(random.uniform(10, 59), 6)

        # Instantiate a realistic OpenAI RateLimitError object
        fake_error_message = {
            "error": {
                "message": f"Quota exceeded. Please retry in {delay_time}s."
            }
        }
        rate_limit_exception = RateLimitError(
            message="Rate limit hit",
            response=mock_http_response,
            body=fake_error_message
        )
        
        # Use side_effect to raise the error when the mock API is called
        self.mock_client.chat.completions.create.side_effect = rate_limit_exception
        
        # Run the code block containing your try/except catchers
        result = self.model.get_response(prompt="Retry test", row_index=random.randint(0, 10))
        
        # Assertions: Verify extraction and fallback safeties
        self.assertEqual(result, expected_result)  # Your block returns {} on RateLimitError exhaustion

        # Assert mock_sleep was called twice, specifically with delay_time each time
        mock_sleep.assert_has_calls([call(delay_time), call(delay_time)])

        # Optionally verify total call count
        self.assertEqual(mock_sleep.call_count, 2)

class TestGetDelayTime(unittest.TestCase):
    def setUp(self):
        """Runs automatically before EVERY test method to ensure isolated state."""
        self.chat_model = ChatProcessorModel(random.randint(10, 40))
        self.delay_time = random.uniform(5, 59)
        self.static_error_message = f"""🚨 You exceeded your current quota, please check your plan and billing details. For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. To monitor your current usage, head to: https://ai.dev/rate-limit. 
        * Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, model: {src.constants.MODEL_NAME}
        Please retry in {self.delay_time}s. 🚨""".strip()

    def test_exact_google_error(self):
        """Should accurately parse long floating point variants with a trailing period."""
        msg = "🚨 Quota exceeded... Please retry in 36.07146633s. 🚨"
        self.assertEqual(self.chat_model._parse_delay_time(msg), 36.07146633)

    def test_integer_delay(self):
        """Should handle errors that pass clean integers."""
        msg = "Resource exhausted. Please retry in 15s."
        self.assertEqual(self.chat_model._parse_delay_time(msg), 15.0)

    def test_missing_trailing_period(self):
        #delay_time = random.uniform(10, 59)
        print(f"test: delay_time={self.delay_time}")
        """
        Should work perfectly even if the string drops the period after the 's'.
        Note: We usually don't test protected methods but will make an exception for this case.
        """
        msg = f"Rate limit reached. Please retry in {self.delay_time}s"
        actual = self.chat_model._parse_delay_time(msg)
        expected = self.delay_time
        self.assertEqual(actual, expected)

    def test_missing_anchor(self):
        """Should gracefully fall back to default timer if anchor string is absent."""
        msg = "Internal Server Error: Connection timed out unexpectedly."
        self.assertEqual(self.chat_model._parse_delay_time(msg), RATE_LIMIT_PAUSE_TIMER)
    
    def test_long_delay_time(self):
        """Should return default delay time of 30 seconds if vendor delay time is too long"""
        delay_time = 180
        msg = f"Rate limit reached. Please retry in {delay_time}s"
        self.assertEqual(self.chat_model._parse_delay_time(msg), RATE_LIMIT_PAUSE_TIMER)

if __name__ == "__main__":
    unittest.main()
