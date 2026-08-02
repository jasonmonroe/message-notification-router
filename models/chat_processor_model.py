# code/models/chat_processor_model.py

# Python Libraries
import json

# Vendor Libraries
from openai import OpenAI
from openai.types.chat import ChatCompletion

# Local Libraries
from src.constants import MODEL_API_KEY, MODEL_API_URL, MODEL_NAME
from src.utils import show_banner

class ChatProcessorModel:
    def __init__(self, dataset:dict):
        self.name = "WhatsApp Processing Model"
        
        self._client = self._load_model(dataset.get("row_total", 0))

    def _load_model(self, row_total:int):
        if MODEL_API_URL is None or MODEL_NAME is None or MODEL_API_KEY is None:
            print("⚠️ Credentials aren't properly being read. Check .env file. ⚠️")
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


    def get_response(self) -> dict:
        pass


    def _filter_response(self) -> str:
        pass
