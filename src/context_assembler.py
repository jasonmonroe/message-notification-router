# src/context_assembler.py

# Python Libraries
import cv2
import os
import pandas as pd
import pytesseract
import whisper
 

# Local Libraries
from src.constants import ROUTING_PROMPT_TEMPLATE


class ContextAssembler:
    """
    This class takes all the data and filters by each message row.  Then if needed it gets the media information to 
    provide additional conte

    Last it builds a prompt with all relevant data garned by the routed logic.
    """
    def __init__(self, data) -> None:
        self._business_id = None
        self._row = None
        
        # Reference master dataframes from DataHandler
        self._users = data.users
        self._groups = data.groups
        self._group_members = data.group_members
        self._business_accounts = data.business_accounts
        self._message_history = data.message_history
        self._user_business_history = data.user_business_history
        self._images = data.images
        self._voice_notes = data.voice_notes

    def get_prompt(self, dataset: dict) -> str:

        # Route messages
        return ROUTING_PROMPT_TEMPLATE.format(
            business_id=self._business_id if pd.notna(self._business_id) else "N/A",
            business_sender_context=self._format_business_sender_context(dataset.get("business_accounts"), dataset.get("user_business_history")),
            group_id=self._row.group_id,
            group_metadata_context=self._format_group_metadata(dataset.get("group_members")),
            historical_evidence=self._format_historical_evidence(dataset.get("message_history")),
            incoming_message_context=self._format_incoming_message(),
            media_content_description=self._format_media_content_description(dataset.get("filepath")),
            media_filename=os.path.basename(dataset.get("filepath")),
            media_type=self._row.media_type,
            message_id=self._row.message_id,
            recipient_user_context=self._format_recipient_user_context(dataset.get("users")),
            user_id=self._row.user_id,
        )

    def _format_incoming_message(self) -> str:
        row = self._row
        return f"""
- Message ID: {row.message_id}
- Conversation Type: {row.conversation_type}
- Group ID: {row.group_id if pd.notna(row.group_id) else 'None'}
- Sender / Business ID: {row.business_id if pd.notna(row.business_id) else (row.sender_user_id if pd.notna(row.sender_user_id) else 'None')}
- Target User ID: {row.user_id}
- Created At: {row.created_at}
- Message Text: {row.message_text if pd.notna(row.message_text) else '[Media Message]'}
- Media Type: {row.media_type if pd.notna(row.media_type) else 'None'}
- Forwarded Count: {row.forwarded_count}
        """.strip()

    def _format_media_content_description(self, media_filepath: str | None) -> str:
        # For Voice Notes / Audio: 
        # Pass the transcription of the audio file (if you have an automatic speech-to-text 
        # step or dataset column) so the model can read what was said.

    	# For Images: 
        # Pass a caption, OCR text extracted from the image, or a short description of the image content.

        if self._row.media_type == "image":   
            description = self._extract_image_text(media_filepath)
    
        elif self._row.media_type == "voice":
            description = self._transcribe_audio_file(media_filepath)
        else:
            description = "No media content available."

        return description.strip()

    def _format_recipient_user_context(self, user_df: pd.DataFrame) -> str:
        if user_df is None or user_df.empty:
            return "No specific user profile data found."
        
        user_row = user_df.iloc[0]

        return f"""
- User ID: {user_row.get('user_id', self._row.user_id)}
- Do Not Disturb Window: {user_row.get('do_not_disturb_window', 'None')}
- 30-Day Stats: {user_row.get('messages_opened_30d', 0)} opened, {user_row.get('messages_replied_30d', 0)} replied, {user_row.get('notifications_dismissed_30d', 0)} dismissed, {user_row.get('messages_reported_30d', 0)} reported.
        """.strip()

    def _format_business_sender_context(self, business_df: pd.DataFrame, business_history_df: pd.DataFrame) -> str:
        # If the current message has no business ID, it's a personal or group message. Skip business context entirely!
        if pd.isna(self._business_id) or not self._business_id:
            return "Not applicable (Personal or Group message)."
        
        business_info = ""
        if business_df is not None and not business_df.empty:
            b_row = business_df.iloc[0]
            business_info = f"""
- Business Name: {b_row.get('brand_name', self._business_id)}
- Category: {b_row.get('category', 'Unknown')}
- Verified: {b_row.get('is_verified', 'Unknown')}
- Sender Domain: {b_row.get('domain', 'Unknown')}
            """.strip()
        else:
            business_info = f"- Business ID: {self._business_id} (No extended metadata found)"

        history_info = ""
        if business_history_df is not None and not business_history_df.empty:
            history_rows = "\n".join([
                f"  * Interaction: {row.get('interaction_type', 'unknown')} at {row.get('created_at', 'unknown')}"
                for _, row in business_history_df.iterrows()
            ])
            history_info = f"\n- User-Business Interaction History:\n{history_rows}"

        return f"{business_info}\n{history_info}".strip()

    def _format_historical_evidence(self, history_df: pd.DataFrame) -> str:
        if history_df is None or history_df.empty:
            return "No relevant historical message evidence found."
        
        # Take up to 5 recent historical messages as evidence context
        recent_history = history_df.head(5)
        evidence_lines = []
        for _, row in recent_history.iterrows():
            evidence_lines.append(
                f"- ID: {row.get('message_id')} | Date: {row.get('created_at')} | Text: {str(row.get('message_text', ''))[:80]}..."
            )
        return "\n".join(evidence_lines)
        

    def _format_group_metadata(self, group_members: pd.DataFrame) -> str:
        # Group by group ID.
        group_df = self._groups[self._groups["group_id"] == self._row.group_id]

        # Get specific group membership row for this user & group
        group_members_row = group_members[
            (group_members["group_id"] == self._row.group_id) & 
            (group_members["user_id"] == self._row.user_id)
        ]

        if group_members_row.empty:
            return "No group membership data found for this user."

        group_muted_by_user = group_members_row["group_muted_by_user"].iloc[0]
        role = group_members_row["role"].iloc[0]
        group_name = group_df.iloc[0].get("group_name", self._row.group_id) if not group_df.empty else self._row.group_id
        group_type = group_df.iloc[0].get("group_type", "unknown") if not group_df.empty else "unknown"

        return f"""
- Group ID: {self._row.group_id}
- Group Name: {group_name}
- Group Type: {group_type}
- User Role: {role}
- Muted by User: {'Muted' if group_muted_by_user == 1 else 'Unmuted'}
        """.strip()
            
    def build_prompt_by_user(self, message_row: pd.DataFrame) -> str:
        """
        Ingestion and Context
        The Ingestion & Context Pipeline: For each row in messages.csv, aggregate the corresponding user 
        preferences (users.csv), group role (group_members.csv), business status (business_accounts.csv), 
        and past chat history (message_history.csv) into a single structured context payload.
        """
        self._row = message_row
        self._business_id = self._row.business_id

        # Get filtered dataset to usein the prompt generation
        dataset = self._filter_by_user(self._row.user_id)
        filepath = self._get_media_filepath(self._row.media_type, self._row.media_id)
    
        if filepath:
            dataset["filepath"] = filepath

        # Generate prompt string
        prompt = self.get_prompt(dataset)
        #dataset["prompt"] = prompt
        #
        #return dataset
        return prompt

    def _get_media_filepath(self, media_type: str | None, media_id: str | None) -> str | None:
        print(f"DBG: _get_media_filepath() media_type={media_type}, media_id={media_id}")
        if pd.isna(media_type) or not media_type or pd.isna(media_id) or not media_id:
            return None

        if media_type == "image":
            media_df = self._images
            id_column = "image_id" 
        elif media_type == "voice":
            media_df = self._voice_notes
            id_column = "voice_note_id"   
        else:
            return None

        if media_df is not None and not media_df.empty and id_column in media_df.columns:
            match = media_df[media_df[id_column] == media_id]
            
            if not match.empty:
                raw_path = match.iloc[0].get("file_path") or match.iloc[0].get("filename")
                if raw_path:
                    # Ensure path points to the correct location (e.g., inside dataset directory)
                    # Adjust 'dataset/' prefix if your folder structure differs
                    full_path = raw_path if os.path.exists(raw_path) else os.path.join("dataset", raw_path)
                    return full_path

        return None

    # @TODO - old version
    def _get_media_filepath2(self, media_type: str | None, media_id: str | None) -> str | None:
        print(f"media_type={media_type}, media_id={media_id}")
        if pd.isna(media_type) or not media_type or pd.isna(media_id) or not media_id:
            return None

        if media_type == "image":
            media_df = self._images
            id_column = "image_id" 
        elif media_type == "voice":
            media_df = self._voice_notes
            id_column = "voice_note_id"   
        else:
            return None

        # Use id_column variable here instead of hardcoding "media_id"
        if media_df is not None and not media_df.empty and id_column in media_df.columns:    
            match = media_df[media_df[id_column] == media_id]
            
            if not match.empty:
                return match.iloc[0].get("file_path") or match.iloc[0].get("filename")

        return None

    def _filter_by_user(self, user_id: str) -> dict:
        # Filter user business history
        user_business_history = self._user_business_history[self._user_business_history["user_id"] == user_id]
        business_ids = user_business_history["business_id"].tolist()   
        self._business_id = self._row.business_id
   
        return {
            "users": self._users[self._users["user_id"] == user_id],
            "group_members": self._group_members[self._group_members["user_id"] == user_id],
            "business_accounts": self._business_accounts[self._business_accounts["business_id"].isin(business_ids)] if business_ids else pd.DataFrame(),
            "message_history": self._message_history[self._message_history["user_id"] == user_id],
            "user_business_history": user_business_history,
        }

    def _transcribe_audio_file(self, audio_filepath: str) -> str:
        """
        Loads a local audio file and transcribes it into text using OpenAI Whisper.
        """
        try:
            # Load the base model (options: tiny, base, small, medium, large)
            model = whisper.load_model("base")
            
            # Transcribe the audio file path
            result = model.transcribe(audio_filepath)
            
            return "- Audio Transcription: " + result["text"]
        
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


    def export(self) -> dict:
        return {}
