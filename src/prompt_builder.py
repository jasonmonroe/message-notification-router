# src/prompt_builder.py

# Vendor Libraries
import os
import pandas as pd
import textwrap

# Local Libraries
from src.constants import MAX_HISTORICAL_MESSAGES, PEP8_LINE_LEN, ROUTING_PROMPT_TEMPLATE
from src.context_assembler import ContextAssembler


class PromptBuilder:
    def __init__(self, dataset: dict):
        """
        Builds full prompt by including placeholder strings in the appropriate areas.

        """

        """
        filtered dataset:
            "business_accounts" 
            "group_members" 
            "message_history" 
            "users":  
            "user_business_history": 
        """


        # message_id,user_id,conversation_type,group_id,business_id,sender_user_id,created_at,message_text,media_type,media_id,forwarded_count
        #: # => filtered dataset
        #: ## => message row column
        #: ### => message row
        #: temp variables

        self.business_accounts = None #
        #self.business_id = None ##
        #self.group_id = None ##
        self.group_members = None #
        self.media_description = None
        self.media_filename = None      # <== image_id or voice_note_id
        self.media_filepath = None #
        #self.media_type = None ##
        self.message_history = None #
        #self.message_id = None ##
        self.message = None ### message row
        self.prompt = ""
        self.user_business_history = None #
        #self.user_id = None ##
        self.users = None #

        self._set_attrs(dataset)

        self.prompt = self.build(dataset.get("groups"))

    def _set_attrs(self, dataset: dict) -> None:
        for key, value in dataset.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def build(self, 
        #business_accounts, 
        #group_members,
        groups: pd.DataFrame, 
        #media_filepath, 
        #message_history, 
        #user_business_history,
        #users,
        ):

        self.prompt = ROUTING_PROMPT_TEMPLATE.format(
            business_id=self.message.business_id,
            business_sender_context=self._format_business_sender(),
            group_id=self.message.group_id,
            group_metadata_context=self._format_group_metadata(groups),
            historical_evidence=self._format_historical_evidence(self.message_history),
            incoming_message_context=self._format_incoming_message(),
            media_content_description=self.media_description,
            media_filename=os.path.basename(self.media_filepath),
            media_type=self.message.media_type,
            #message=self.message,
            message_id=self.message.message_id,
            recipient_user_context=self._format_recipient_user_context(),
            user_id=self.message.user_id,
        )

        """
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
        """


    # --- Format Attributes --- #

    def _format_business_sender(self) -> str:
        # If the current message has no business ID, it's a personal or group message. Skip business context entirely!
        if pd.isna(self.business_id) or not self.business_id:
            return "Not applicable (Personal or Group message)."

        business_df = self.business_accounts
        business_history_df = self.user_business_history

        def _get_business_info(business_df: pd.DataFrame) -> str:
            business_info = ""

            if business_df is not None and not business_df.empty:
                business_row = business_df.iloc[0]
                business_info = textwrap.dedent(f"""\
                    - Business Name: {business_row.get('brand_name', self.message.business_id)}
                    - Category: {business_row.get('category', 'Unknown')}
                    - Verified: {business_row.get('is_verified', 'Unknown')}
                    - Sender Domain: {business_row.get('domain', 'Unknown')}\
                """).strip()
            else:
                business_info = f"- Business ID: {self.message.business_id} (No extended metadata found)"

            return business_info

        def _get_history_info(business_history_df: pd.DataFrame) -> str:
            history_info = ""

            if business_history_df is not None and not business_history_df.empty:
                history_rows = "\n".join([
                    " * Interaction: {interaction_type} at {created_at}".format(
                        interaction_type=self.message.get("interaction_type", "unknown"),
                        created_at=self.message.get("created_at", "unknown")
                    ).strip()
                    for _, row in business_history_df.iterrows()
                ])

                history_info = f"\n- User-Business Interaction History:\n{history_rows}"

            return history_info

        business_info = _get_business_info(business_df)
        history_info = _get_history_info(business_history_df)    

        return f"{business_info}\n{history_info}".strip()


    def _format_group_metadata(self, groups_df: pd.DataFrame) -> str:
        # Filter for the target group
        #group_df = groups[groups["group_id"] == self.group_id]

        #group_members = self.group_members
        
        # Filter for the specific user's membership in that group
        # @TODO - is this even necessary?
        group_members_row = self.group_members[
            (self.group_members["group_id"] == self.message.group_id) & 
            (self.group_members["user_id"] == self.message.user_id)
        ]
        
        print(f"DBG: group_members_row = {group_members_row}")

        if group_members_row.empty:
            return "No group membership data found for this user."
            
        # Extract user-specific data
        group_muted_by_user = group_members_row["group_muted_by_user"].iloc[0]
        muted_by_user = "Muted" if group_muted_by_user == 1 else "Unmuted"
        role = group_members_row["role"].iloc[0]
        
        # Safely extract group-specific metadata
        group_name = groups_df["group_name"].iloc[0] if not groups_df.empty and "group_name" in groups_df else self.message.group_id
        group_type = groups_df["group_type"].iloc[0] if not groups_df.empty and "group_type" in groups_df else "unknown"
        
        # Formats as a clean, vertical, token-optimized markdown list
        return textwrap.dedent(f"""\
            - Group ID: {self.message.group_id}
            - Group Name: {group_name}
            - Group Type: {group_type}
            - User Role: {role}
            - Muted by User: {muted_by_user}\
        """).strip()

    def _format_historical_evidence(self) -> str:
        if self.message_history is None or self.message_history.empty:
            return "No relevant historical message evidence found."
        
        # Take up to 5 recent historical messages as evidence context
        recent_history = self.message_history.head(MAX_HISTORICAL_MESSAGES)

        evidence_lines = []
        for _, row in recent_history.iterrows():
            evidence_lines.append(
                "- ID: {message_id} | Date: {created_at} | Text: {message_text}...".format(
                    message_id=row.get("message_id"),
                    created_at=row.get("created_at"),
                    message_text=str(row.get("message_text", ""))[:{PEP8_LINE_LEN}]
                ).strip()
            )

        return "\n".join(evidence_lines)

    def _format_incoming_message(self) -> str:
        group_id = self.message.group_id if pd.notna(self.message.group_id) else "None"
        sender_or_business_id = self.message.business_id if pd.notna(self.message.business_id) else (self.message.sender_user_id if pd.notna(self.message.sender_user_id) else "None")
        message_text = self.message.message_text if pd.notna(self.message.message_text) else "[Media Message]"
        media_type = self.message.media_type if pd.notna(self.message.media_type) else "None"

        return textwrap.dedent("""\
            - Message ID: {message_id}
            - Conversation Type: {conversation_type}
            - Group ID: {group_id}
            - Sender / Business ID: {sender_or_business_id}
            - Target User ID: {user_id}
            - Created At: {created_at}
            - Message Text: {message_text}
            - Media Type: {media_type}
            - Forwarded Count: {forwarded_count}\            
        """).format(
                message_id=self.message.message_id,
                conversation_type=self.message.conversation_type,
                group_id=group_id,
                sender_id_or_business_id=sender_or_business_id,
                user_id=self.message.user_id,
                created_at=self.message.created_at,
                message_text=message_text,
                media_type=media_type,
                forwarded_count=self.message.forwarded_count,
        ).strip()

    def _format_recipient_user_context(self) -> str:
        if self.users is None or self.users.empty:
            return "No specific user profile data found."
        
        user_row = self.users.iloc[0]

        return textwrap.dedent("""\
            - User ID: {user_id}
            - Do Not Disturb Window: {do_not_disturb_window}
            - 30-Day Stats: {messages_opened_30d} opened, {messages_replied_30d} replied, {notifications_dismissed_30d} dismissed, {messages_reported_30d} reported.\
        """).format(
            do_not_disturb_window=user_row.get("do_not_disturb_window", "None"),
            message_opened_30d=user_row.get("messages_opened_30d", 0),
            messages_replied_30d=user_row.get("messages_replied_30d", 0),
            messages_reported_30d=user_row.get("messages_reported_30d", 0),
            notifications_dismissed_30d=user_row.get("notifications_dismissed_30d", 0),
            user_id=user_row.get("user_id", self.message.user_id),
        ).strip()
