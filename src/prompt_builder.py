# src/prompt_builder.py

# Vendor Libraries
import os
import pandas as pd

# Local Libraries
from src.constants import ROUTING_PROMPT_TEMPLATE


class PromptBuilder:
    def __init__(self, dataset: dict):
        self.business_accounts = None
        self.business_id = None
        self.filepath = None
        self.group_id = None
        self.group_members = None
        self.media_filename = None
        self.media_type = None
        self.message_history = None
        self.message_id = None
        self.prompt = ""
        self.user_business_history = None
        self.user_id = None
        self.users = None

        self._set_attrs(dataset)

    def _set_attrs(self, dataset: dict) -> None:
        for key, value in dataset.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def build(self):
        self.prompt = ROUTING_PROMPT_TEMPLATE.format(
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


    # --- Format Attributes --- #
    def _format():
        pass

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


