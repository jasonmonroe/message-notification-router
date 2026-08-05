# src/context_assembler.py

# Python Libraries
import cv2
import os
import pandas as pd
import pytesseract
import whisper

# Local Libraries
from src.prompt_builder import PromptBuilder

class ContextAssembler:
    """
    This class takes all the data and filters by each message row.  Then if needed it gets the media information to 
    provide additional context.
    
    We store the data files on instantiation and then filter by user per message.

    Last it builds a prompt with all relevant data garned by the routed logic.
    """
    def __init__(self, data) -> None:
        self._business_accounts = data.business_accounts
        self._business_id = None
        self._group_members = data.group_members
        self._groups = data.groups
        self._images = data.images
        self._message_history = data.message_history
        self._message = pd.DataFrame() 
        self._user_business_history = data.user_business_history
        self._users = data.users
        self._voice_notes = data.voice_notes

    def build_prompt_by_user(self, message_row: pd.DataFrame) -> str:
        """
        Ingestion and Context
        The Ingestion & Context Pipeline: For each row in messages.csv, aggregate the corresponding user 
        preferences (users.csv), group role (group_members.csv), business status (business_accounts.csv), 
        and past chat history (message_history.csv) into a single structured context payload.
        """

        # Get filtered dataset to usein the prompt generation
        filtered_dataset = self._filter_by_user(message_row.user_id)

        # Filter by group ID, note: group_members has already been filtered by user so we just need to filter by group_id
        filtered_group_dataset = self._filter_by_group(filtered_dataset["group_members"], message_row.group_id)
        filtered_dataset = {**filtered_dataset, **filtered_group_dataset}
        filtered_dataset["message"] = message_row
        
        print(f"DBG: filtered_dataset={filtered_dataset}")

        # Filter by media ID
        filepath = self._get_media_filepath(message_row.media_type, message_row.media_id)
        if filepath:
            filtered_dataset["media_filepath"] = filepath
            filtered_dataset["media_description"] = self._get_media_description(message_row.media_type, filepath)

        # Load Prompt Builder to get the prompt
        builder = PromptBuilder(filtered_dataset)

        print(f"prompt = {builder.prompt}")

        import sys
        sys.exit(0)

        return builder.prompt

    def _filter_by_user(self, user_id: str) -> dict:
        # Filter user business history
        user_business_history = self._user_business_history[self._user_business_history["user_id"] == user_id]
        user_business_ids = user_business_history["business_id"].tolist()   
        business_accounts = self._business_accounts[self._business_accounts["business_id"].isin(user_business_ids)] if user_business_ids else pd.DataFrame()
   
        return { 
            "business_accounts": business_accounts,
            "group_members": self._group_members[self._group_members["user_id"] == user_id],
            "message_history": self._message_history[self._message_history["user_id"] == user_id],
            "user_business_history": user_business_history,
            "users": self._users[self._users["user_id"] == user_id]
        }

    def _filter_by_group(self, group_members: pd.DataFrame, group_id: str) -> dict:
        return {
            "group_members": group_members[group_members["group_id"] == group_id],
            "groups": self._groups[self._groups["group_id"] == group_id],
        }

    def _get_media_filepath(self, media_type: str, media_id: str) -> str | None:
        """
        Get full media filepath from the root directory "dataset" based on message row columns: media_type and media_id
        """

        print(f"DBG: _get_media_filepath() media_type={media_type}, media_id={media_id}")

        # Check columns for missing information return None if missing.
        if pd.isna(media_type) or not media_type or pd.isna(media_id) or not media_id:
            return None

        if media_type == "image":
            id_column = "image_id" 
            media_df = self._images
        elif media_type == "voice":
            id_column = "voice_note_id"   
            media_df = self._voice_notes
        else:
            return None

        # If media dataframe is set get media information by media_id
        if media_df is not None and not media_df.empty and id_column in media_df.columns:
            filtered_media = media_df[media_df[id_column] == media_id]
            
            if filtered_media:
                # @TODO - old: raw_path = filtered_media.iloc[0].get("file_path") or filtered_media.iloc[0].get("filename")
                media_filepath = filtered_media.iloc[0].get("file_path")
                
                if media_filepath:
                    # Ensure path points to the correct location (e.g., inside dataset directory)
                    # Adjust 'dataset/' prefix if your folder structure differs
                    return media_filepath if os.path.exists(media_filepath) else os.path.join("dataset", media_filepath)

        return None

    def _get_media_description(self, media_type: str, media_filepath: str) -> str:
        """
        For Voice Notes / Audio: 
            Pass the transcription of the audio file (if you have an automatic speech-to-text 
            step or dataset column) so the model can read what was said.

    	For Images: 
            Pass a caption, OCR text extracted from the image, or a short description of the 
            image content.
        """

        description = "No media content available."

        if media_type == "image":   
            description = self._extract_image_text(media_filepath)
             
        elif media_type == "voice":
            description = self._transcribe_audio_file(media_filepath)

        return description.strip()
        
    def _transcribe_audio_file(self, audio_filepath: str) -> str:
        """
        Loads a local audio file and transcribes it into text using OpenAI Whisper.
        """
        try:
            # Load the base model (options: tiny, base, small, medium, large)
            base_model = whisper.load_model("base")
            
            # Transcribe the audio file path
            result = base_model.transcribe(audio_filepath)
            
            return "- Audio Transcription: " + result["text"].strip()
        
        except Exception as g:
            print(f"❌ Error transcribing audio: {g}")
            print("Returning empty string!")
            return ""

    def _extract_image_text(self, image_path: str) -> str:
        if not image_path or not os.path.exists(image_path):
            return "Image file not found."

        # Load image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return "Could not load image file."
        
        # Preprocess with OpenCV to improve OCR accuracy
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        
        # Pass the OpenCV-processed image directly into Tesseract
        extracted_text = pytesseract.image_to_string(thresh).strip()
        
        if len(extracted_text) > 0:
            return "- Extracted Image Text (OCR): " + extracted_text
        
        # Fallback if the image contains no readable text (e.g. standard photo)
        return "- Image Description: Attached photograph/image file (No embedded text detected)."
