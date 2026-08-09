# src/context_assembler.py

# +-----------------------------------------------------------------------------+
# |                              CONTEXT ASSEMBLER                              |
# +-----------------------------------------------------------------------------+

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
        self._group_members = data.group_members
        self._groups = data.groups
        self._images = data.images
        self._message_history = data.message_history
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
        filtered_dataset = self._filter_by_user(message_row.user_id, message_row.business_id)

        # Filter by group ID, note: group_members has already been filtered by user so we just need to filter by group_id
        filtered_group_dataset = self._filter_by_group(filtered_dataset["group_members"], message_row.group_id)
        filtered_dataset |= filtered_group_dataset
        
        # Filter by media ID
        media_filepath = self._get_media_filepath(message_row.media_type, message_row.media_id)
        if media_filepath:
            filtered_dataset["media_filepath"] = media_filepath
            filtered_dataset["media_description"] = self._get_media_description(message_row.media_type, media_filepath)

        filtered_dataset["message"] = self._clean_message(message_row)

        # Load Prompt Builder to get the prompt
        builder = PromptBuilder(filtered_dataset)

        return builder.prompt

    def _filter_by_user(self, user_id: str, business_id: str) -> dict:
        # Filter user business history by the user and the business.
        user_business_history = self._user_business_history[self._user_business_history["user_id"] == user_id]
        message_history = self._message_history[self._message_history["user_id"] == user_id]
        business_accounts = self._business_accounts

        if pd.notna(business_id) and business_id:
            user_business_history = user_business_history[user_business_history["business_id"] == business_id]

            # Now that we have filtered the user_business_history by user_id and business_id get the business accounts 
            # this user has interacted with.
            business_accounts = self._business_accounts[self._business_accounts["business_id"] == business_id]
            message_history = message_history[message_history["business_id"] == business_id]
        
        return { 
            "business_accounts": business_accounts,
            "group_members": self._group_members[self._group_members["user_id"] == user_id],
            "message_history": message_history,
            "user_business_history": user_business_history,
            "users": self._users[self._users["user_id"] == user_id]
        }

    def _filter_by_group(self, group_members: pd.DataFrame, group_id: str) -> dict:
        return {
            "group_members": group_members[group_members["group_id"] == group_id],
            "groups": self._groups[self._groups["group_id"] == group_id],
        }

    def _clean_message(self, message_row: pd.DataFrame) -> pd.DataFrame:
        message_dict = message_row._asdict()

        # Clean out the nan values, replacing them with empty strings
        cleaned_dict = {
            key: ("" if pd.isna(value) else value) 
            for key, value in message_dict.items()
        }

        return cleaned_dict

    def _get_media_filepath(self, media_type: str, media_id: str) -> str | None:
        """
        Get full media filepath from the root directory "dataset" based on message row columns: media_type and media_id
        """

        # Check columns for missing information return None if missing.
        if pd.isna(media_type) or not media_type or pd.isna(media_id) or not media_id:
            return None

        if media_type == "image":
            id_column, media_df = "image_id", self._images 
        elif media_type == "voice":
            id_column, media_df = "voice_note_id", self._voice_notes
        else:
            return None

        # If media dataframe is set get media information by media_id
        if media_df is not None and not media_df.empty and id_column in media_df.columns:
            filtered_media = media_df[media_df[id_column] == media_id]
            
            if not filtered_media.empty:
                media_filepath = filtered_media.iloc[0].get("file_path")
                
                # Ensure path points to the correct location (e.g., inside dataset directory)
                # Adjust 'dataset/' prefix if your folder structure differs
                if media_filepath:
                    return media_filepath if os.path.exists(media_filepath) else os.path.join("dataset", media_filepath)

        return None

    def _get_media_description(self, media_type: str, media_filepath: str) -> str | None:
        """
        For Voice Notes / Audio: 
            Pass the transcription of the audio file (if you have an automatic speech-to-text 
            step or dataset column) so the model can read what was said.

    	For Images: 
            Pass a caption, OCR text extracted from the image, or a short description of the 
            image content.

        If none, remove it entirely otherwise get the description and build the context around it.
        """

        description = None

        if media_type == "image":   
            description = self._extract_image_text(media_filepath)
             
        elif media_type == "voice":
            description = self._transcribe_audio_file(media_filepath)

        return description

    def _transcribe_audio_file(self, audio_filepath: str) -> str:
        """
        Loads a local audio file and transcribes it into text using OpenAI Whisper.
        """
        try:
            # Load the base model (options: tiny, base, small, medium, large)
            base_model = whisper.load_model("base")
            
            # Transcribe the audio file path
            result = base_model.transcribe(audio_filepath)
            
            return "Audio Transcription: " + result["text"].strip()
        
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
            return "Extracted Image Text (OCR): " + extracted_text
        
        # Fallback if the image contains no readable text (e.g. standard photo)
        return "Image Description: Attached photograph/image file (No embedded text detected)."
