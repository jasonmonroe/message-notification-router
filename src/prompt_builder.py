# src/prompt_builder.py

# +-----------------------------------+
# |          PROMPT BUILDER           |
# +-----------------------------------+

# Vendor Libraries
import os
import pandas as pd
import textwrap
import sys

# Local Libraries
from src.constants import MAX_HISTORICAL_MESSAGES, ROUTING_PROMPT_TEMPLATE

class PromptBuilder:
    def __init__(self, dataset: dict):
        """
        Builds full prompt by including placeholder strings in the appropriate areas.

        filtered dataset:
            - business_accounts
            - group_members
            - message_history
            - users
            - user_business_history 
        """

        self.business_accounts = None
        self.group_members = None
        self.media_description = None
        self.media_filepath = None
        self.message_history = None
        self.message = None
        self.prompt = ""
        self.user_business_history = None
        self.users = None

        self._set_attrs(dataset)

        self.prompt = self.build(dataset.get("groups"))

    def _set_attrs(self, dataset: dict) -> None:
        for key, value in dataset.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def build(self, groups: pd.DataFrame) -> str:
        return ROUTING_PROMPT_TEMPLATE.format(            
            business_sender_context=self._format_business_sender_xml(),
            group_metadata_context=self._format_group_metadata_xml(groups),
            historical_evidence=self._format_historical_evidence_xml(),
            incoming_message_context=self._format_incoming_message_xml(),
            media_context=self._format_media_context_xml(),
            message_id=self.message.get("message_id"),
            recipient_user_context=self._format_recipient_user_context_xml(),
        )

    # --- Format Attributes --- #

    def _format_incoming_message_xml(self) -> str:
        group_id = self.message.get("group_id") if pd.notna(self.message.get("group_id")) else "None"
        sender_or_business_id = self.message.get("business_id") if pd.notna(self.message.get("business_id")) else (self.message.get("send_user_id") if pd.notna(self.message.get("send_user_id")) else "None")
        message_text = self.message.get("message_text") if pd.notna(self.message.get("message_text")) else "[Media Message]"
        media_type = self.message.get("media_type") if pd.notna(self.message.get("media_type")) else "None"

        return """
        <incoming_message id="{message_id}" conversation_type="{conversation_type}" group_id="{group_id}" sender_id="{sender_or_business_id}" target_user_id="{user_id}" created_at="{created_at}" media_type="{media_type}" forwarded_count="{forwarded_count} ">
            <message_text>{message_text}</message_text>
        </incoming_message>
        """.strip().format(
            message_id=self.message.get("message_id"),
            conversation_type=self.message.get("conversation_type"),
            group_id=group_id,
            sender_or_business_id=sender_or_business_id,
            user_id=self.message.get("user_id"),
            created_at=self.message.get("created_at"),
            message_text=message_text,
            media_type=media_type,
            forwarded_count=self.message.get("forwarded_count"),
        )

    def _format_business_sender_xml(self) -> str:
        # If the current message has no business ID, it's a personal or group message. Skip business context entirely!
        if pd.isna(self.message.get("business_id")) or not self.message.get("business_id"):
            return ""
        
        business_df = self.business_accounts
        business_history_df = self.user_business_history

        def _get_business_info_xml_open(business_df: pd.DataFrame) -> str:
            business_info = ""
            if business_df is not None and not business_df.empty:
                business_row = business_df.iloc[0]

                business_info = """
                <business_sender id="{business_id}" name="{name}" category="{category}" verified="{verified}" sender_domain="{sender_domain}">
                <interactions>
                """.strip().format(
                    business_id=self.message.get("business_id"),
                    name=business_row.get("display_name", self.message.get("brand_name")),
                    category=business_row.get("category"),
                    verified="true" if business_row.get("verified") == 1 else "false",
                    sender_domain=business_row.get("domain_used_by_sender", "Unknown"),
                )

            return business_info

        def _get_history_info_xml(business_history_df: pd.DataFrame) -> str:
            history_info_xml = ""

            if business_history_df is not None and not business_history_df.empty:

                # Filter history for this specific business/user if needed, 
                # and dynamically check for whatever columns actually exist in your dataframe.
                history_rows = []
                for _, row in business_history_df.iterrows():
                    # Use real columns present in your dataset rows: action, message_type, created_at     
                    xml_content = """<interaction user_id="{user_id}" relation="{why_user_knows_account}" last_reply_at="{last_reply_at}" last_activity="{last_activity_at}"/>""".strip().format(
                        last_activity_at=row.get("last_activity_at") if row.get("last_activity_at") else "unknown",
                        last_reply_at=row.get("last_reply_at") if row.get("last_reply_at") else "unknown",
                        user_id=row.get("user_id"),
                        why_user_knows_account=row.get("why_user_knows_account"),
                    )
                
                    # Format each row as a clean self-closing XML tag
                    history_rows.append(xml_content)
                    
                history_info_xml = "\n".join(history_rows)
           
            return history_info_xml


        business_info_xml_open = _get_business_info_xml_open(business_df)
        history_info_xml = _get_history_info_xml(business_history_df) # interactions    
        business_info_xml_close = "</interactions></business_sender>"

        return f"""
        {business_info_xml_open}
            {history_info_xml}
        {business_info_xml_close}""".strip() 

    def _format_group_metadata_xml(self, groups_df: pd.DataFrame) -> str:
        # Filter for the target group
        group_members_row = self.group_members[
            (self.group_members["group_id"] == self.message.get("group_id")) & 
            (self.group_members["user_id"] == self.message.get("user_id"))
        ]
        
        # No group membership data found for this user.
        if group_members_row.empty:
            return ""
            
        # Extract user-specific data
        group_muted_by_user = group_members_row["group_muted_by_user"].iloc[0]
        muted_by_user = "Muted" if group_muted_by_user == 1 else "Unmuted"
        group_user_role = group_members_row["role"].iloc[0]
        
        # Safely extract group-specific metadata
        group_name = groups_df["group_name"].iloc[0] if not groups_df.empty and "group_name" in groups_df else self.message.get("group_id")
        group_type = groups_df["group_type"].iloc[0] if not groups_df.empty and "group_type" in groups_df else "unknown"
        
        # Formats as a clean, vertical, token-optimized markdown list
        return """
        <group_metadata id="{group_id}" type="{group_type}" user_role="{group_user_role}" user_mute_status="{group_muted_by_user}">
            <group_name>{group_name}</group_name>
        </group_metadata>
        """.strip().format(
            group_id=self.message.get("group_id"),
            group_name=group_name,
            group_type=group_type,
            group_user_role=group_user_role,
            group_muted_by_user=muted_by_user,
        )

    def _format_historical_evidence_xml(self) -> str:
        # No relevant historical message evidence found.
        if self.message_history is None or self.message_history.empty:
            return ""
        
        # Take up to 5 recent historical messages as evidence context
        recent_history = self.message_history.head(MAX_HISTORICAL_MESSAGES)
 
        xml_open = """<historical_evidence status="optional">""".strip()
        xml_close = "</historical_evidence>"
       
        xml_messages = []
        for _, row in recent_history.iterrows():
            xml_message = """\t<message id="{message_id}" date="{created_at}">{message_text}</message>""".strip().format(
                message_id=row.get("message_id"),
                created_at=row.get("created_at"),
                message_text=str(row.get("message_text", "")).strip()
            )
            xml_messages.append(xml_message)

        xml_content = "\n".join(xml_messages)
        
        return f"{xml_open}{xml_content}{xml_close}"

    def _format_media_context_xml(self) -> str:
        if self.media_description:
            return """
                <media_attachment>
                    <metadata type="{media_type}" filename="{media_filename}" />
                    <description>{media_content_description}</description>
                </media_attachment>
                """.strip().format(
                    media_type=self.message.get("media_type"),
                    media_filename=os.path.basename(self.media_filepath),
                    media_content_description=self.media_description.strip(),
                )

        return ""

    def _format_recipient_user_context_xml(self) -> str:
        # No specific user profile data found.
        if self.users is None or self.users.empty:
            return ""
        
        user = self.users.iloc[0]

        return """
        <recipient_user id="{user_id}" dnd_window="{do_not_disturb_window}">
            <stats thirty_day_opened="{messages_opened_30d}" thirty_day_replied="{messages_replied_30d}" thirty_day_dismissed="{messages_reported_30d}" thirty_day_reported="{notifications_dismissed_30d}" />
        </recipient_user>
        """.strip().format(
            do_not_disturb_window=user.get("do_not_disturb_window", "None"),
            messages_opened_30d=user.get("messages_opened_30d", 0),
            messages_replied_30d=user.get("messages_replied_30d", 0),
            messages_reported_30d=user.get("messages_reported_30d", 0),
            notifications_dismissed_30d=user.get("notifications_dismissed_30d", 0),
            user_id=user.get("user_id", self.message.get("user_id")),
        )
