# code/models/chat_processor_model.py

# Python Libraries
import json

# Vendor Libraries
from openai import OpenAI
from openai.types.chat import ChatCompletion

# Local Libraries
from src.constants import (
    MODEL_API_KEY, 
    MODEL_API_URL, 
    MODEL_NAME, 
    SYSTEM_INSTRUCTIONS
    )
    
from src.utils import show_banner

class ChatProcessorModel:
    def __init__(self, dataset:dict):
        self.name = "WhatsApp Chat Processing Model"
        
        self._client = self._load_model(dataset.get("row_total", 0))

    def _load_model(self, row_total:int) -> OpenAI:

        if MODEL_API_URL is None or MODEL_NAME is None or MODEL_API_KEY is None:
            ValueError("🚨 Credentials aren't properly being read. Check .env file. 🚨")
            return None
             
        subtitles = [
            f"🤖MODEL_NAME: {MODEL_NAME}",
            f"🌐️MODEL_API_URL: {MODEL_API_URL}",
            f"📄️CVS ROWS: {row_total}"
        ]

        show_banner(self.name.upper(), subtitles)
        
        return OpenAI(
            base_url=MODEL_API_URL,
            api_key=MODEL_API_KEY,
            timeout=120,   # ⏱️ Kill the connection if it hangs over 120 seconds
            max_retries=3, # 🔄 Automatically back off and retry 3 times natively
        )


    def get_response(self, prompt: str) -> dict:

        response = self._client.responses.create(
            model=MODEL_NAME,
            #instructions="You are a machine learning expert with extensive knowledge in multimodal prompts for a AI-powered system for WhatsApp that decides which messages deserve immediate attention, which should wait, and which should be muted.",
            instructions=SYSTEM_INSTRUCTIONS,
            input=prompt,
        )

        return self._filter_response(response)


    def _filter_response(self, response) -> str:
        try:
            content_str = response.choices[0].message.content
            if not content_str:
                return {}

            content = json.loads(content_str)

        except (AttributeError, IndexError, json.JSONDecodeError) as e:
            print(f"❌ Error occurred while parsing response: {e}")
            return {}

        return content

