# code/models/chat_processor_model.py

# Python Libraries
import json
import re

# Vendor Libraries
from openai import OpenAI
from openai.types.chat import ChatCompletion, ChatCompletionMessage

# Local Libraries
from src.constants import (
    MAX_TOKENS,
    MODEL_API_KEY, 
    MODEL_API_URL, 
    MODEL_NAME, 
    SYSTEM_INSTRUCTIONS
    )

from src.utils import show_banner

class ChatProcessorModel:
    def __init__(self, row_cnt: int):
        if MODEL_API_URL is None or MODEL_NAME is None or MODEL_API_KEY is None:
            raise ValueError("🚨 Credentials aren't properly being read. Check .env file. 🚨")
             
        subtitles = [
            f"🤖MODEL_NAME: {MODEL_NAME}",
            f"🌐️MODEL_API_URL: {MODEL_API_URL}",
            f"📄️DATA ROWS: {row_cnt}"
        ]

        self.name = "WhatsApp Chat Processing Model"
        show_banner(self.name.upper(), subtitles)
        
        self._client = self._load_model(row_cnt)

    def _load_model(self, row_cnt:int) -> OpenAI:
        return OpenAI(
            base_url=MODEL_API_URL,
            api_key=MODEL_API_KEY,
            timeout=120,   # ⏱️ Kill the connection if it hangs over 120 seconds
            max_retries=3, # 🔄 Automatically back off and retry 3 times natively
        )

    def get_response(self, prompt: str) -> dict:
        """
        Calls OpenAI model with instructions and prompt context and waits for a response.  The response is then
        filtered and returned in a specific format for output.

        https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create
        """
       
        response = self._client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_completion_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            top_p=1.0,
            timeout=90.0
        )

        return self._format_response(self._filter_response(response))

    def _filter_response(self, response: ChatCompletion) -> dict:
        try:
            if hasattr(response, "choices"):
                choice = response.choices[0]
                content_str = choice.message.content
            elif hasattr(response, "content"):
                content_str = response.content
            elif isinstance(response, str):
                content_str = response
            else:
                content_str = str(response)

            if not content_str:
                return {}

            cleaned_str = content_str.strip()

            # Try extracting from markdown code blocks first
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_str, re.DOTALL)
            if json_match:
                cleaned_str = json_match.group(1)
            else:
                # Fallback: Find the first '{' and the last '}'
                start_idx = cleaned_str.find("{")
                end_idx = cleaned_str.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    cleaned_str = cleaned_str[start_idx:end_idx+1]
                elif start_idx != -1:
                    # Auto-repair if the model cut off right at the end before closing '}'
                    cleaned_str = cleaned_str[start_idx:]
                    if not cleaned_str.endswith("}"):
                        cleaned_str += "\n}"

            content = json.loads(cleaned_str.strip())
            return content

        except (AttributeError, IndexError, json.JSONDecodeError) as e:
            print(f"🚨 Error occurred while parsing response: {e} 🚨")
            print(f" `repr(content_str)` was: {repr(content_str)}")
            return {}

    def _format_response(self, response: dict) -> list:
        return list(response.values())
