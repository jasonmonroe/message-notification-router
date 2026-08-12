# src/prompt_builder.py

# +-----------------------------------------------------------------------------+
# |                                PROMPT BUILDER                               |
# +-----------------------------------------------------------------------------+

# Python Libraries
import xml.etree.ElementTree as xml
from xml.dom import minidom

# Vendor Libraries
import os
import pandas as pd

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
        self.daily_notification_summary = None # ?
        self.events = None
        self.group_members = None
        self.media_description = None
        self.media_filepath = None
        self.message_events = None # ?
        self.message_history = None
        self.message = None
        self.notifications = None
        self.prompt = ""
        self.user_business_history = None
        self.users = None

        self._set_attrs(dataset)

        self.prompt = self._build(dataset.get("groups"))

    def _set_attrs(self, dataset: dict) -> None:
        for key, value in dataset.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def _build(self, groups) -> str:
        root = self._open_xml()

        # Build each XML section
        self._users_xml(root)
        self._incoming_message_xml(root)
        self._recipient_user_xml(root)
        self._sender_xml(root)
        self._group_xml(root, groups)
        self._business_xml(root)
        self._user_behavior_profile_xml(root)
        self._media_attachment_xml(root)

        routing_input_xml = self._close_xml(root)

        return ROUTING_PROMPT_TEMPLATE.format(
            routing_input_xml=routing_input_xml
        )

    # --- Format Values (XML) --- #

    def _open_xml(self) -> xml.Element:
        return xml.Element("routing_input")

    def _close_xml(self, root: xml.Element) -> str:
        rough_string = xml.tostring(root, encoding="utf-8")
        reparsed = minidom.parseString(rough_string)
        pretty_xml = reparsed.toprettyxml(indent="  ")

        # Clean up any empty lines generated during pretty printing
        non_empty_lines = [line for line in pretty_xml.splitlines() if line.strip()]
        return "\n".join(non_empty_lines)

    def _users_xml(self, root: xml.Element) -> xml.Element | None:
        sender_id = self.message.get("sender_user_id") if isinstance(self.message, dict) else None
        recipient_user_id = self.message.get("user_id") if isinstance(self.message, dict) else None

        valid_sender = pd.notna(sender_id) and str(sender_id).strip() != ""
        valid_recipient = pd.notna(recipient_user_id) and str(recipient_user_id).strip() != ""

        if not valid_sender and not valid_recipient:
            return None

        users = xml.SubElement(root, "users")
        users.append(xml.Comment(" Basic user notification behavior of sender and reciever "))

        if valid_sender:
            xml.SubElement(users, "user", id=str(sender_id).strip(), role="sender")

        if valid_recipient:
            xml.SubElement(users, "user", id=str(recipient_user_id).strip(), role="recipient")

        return users

    def _incoming_message_xml(self, root: xml.Element) -> xml.Element:
        msg = self.message if isinstance(self.message, dict) else {}
        attrs = {}

        field_map = {
            "id": msg.get("message_id"),
            "conversation_type": msg.get("conversation_type"),
            "group_id": msg.get("group_id"),
            "business_id": msg.get("business_id"),
            "sender_id": msg.get("sender_user_id"),
            "recipient_user_id": msg.get("user_id"),
            "created_at": msg.get("created_at"),
            "media_type": msg.get("media_type"),
            "media_id": msg.get("media_id"),
            "forwarded_count": msg.get("forwarded_count"),
        }

        for attr_name, val in field_map.items():
            if pd.notna(val) and str(val).strip() != "":
                attrs[attr_name] = str(val).strip()

        incoming_message = xml.SubElement(root, "incoming_message", attrs)
        incoming_message.append(xml.Comment(" ⚠️ Crucial: Incoming messages that your system must route "))

        text_val = msg.get("message_text")
        if pd.notna(text_val) and str(text_val).strip() != "":
            xml_text = xml.SubElement(incoming_message, "message_text")
            xml_text.text = str(text_val).strip()

        return incoming_message

    def _recipient_user_xml(self, root: xml.Element) -> xml.Element | None:
        recipient_user_id = self.message.get("user_id") if isinstance(self.message, dict) else None
        
        user_row = None
        if self.users is not None and not self.users.empty:
            user_row = self.users.iloc[0]

        attrs = {}
        if pd.notna(recipient_user_id) and str(recipient_user_id).strip() != "":
            attrs["id"] = str(recipient_user_id).strip()

        if user_row is not None:
            dnd = user_row.get("do_not_disturb_window") if "do_not_disturb_window" in user_row else user_row.get("dnd_window")
            if pd.notna(dnd) and str(dnd).strip() != "":
                attrs["do_not_disturb_window"] = str(dnd).strip()

        recipient_user = xml.SubElement(root, "recipient_user", attrs)
        recipient_user.append(xml.Comment(" Total notification stats of recipient "))

        # Macro Notification Summary (pre-calculated by ContextAssembler)
        if hasattr(self, "notifications") and isinstance(self.notifications, dict) and self.notifications:
            sent = self.notifications.get("total_notifications_sent", 0)
            dismissed = self.notifications.get("total_notifications_dismissed", 0)
            rate = self.notifications.get("overall_dismissal_rate", 0.0)

            if sent > 0 or dismissed > 0:
                macro_elem = xml.SubElement(recipient_user, "macro_notification_summary")
                xml.SubElement(macro_elem, "total_notifications_sent").text = str(int(sent))
                xml.SubElement(macro_elem, "total_notifications_dismissed").text = str(int(dismissed))
                xml.SubElement(macro_elem, "overall_dismissal_rate").text = str(rate)

        # 30-Day Activity Baseline from users dataframe
        if user_row is not None:
            stats_attrs = {}
            for metric in ["opened", "replied", "dismissed", "reported"]:
                col_candidates = [f"thirty_day_{metric}", f"messages_{metric}_30d", f"recent_{metric}"]
                for col in col_candidates:
                    if col in user_row and pd.notna(user_row[col]) and str(user_row[col]).strip() != "":
                        stats_attrs[metric] = str(user_row[col]).strip()
                        break

            if stats_attrs:
                xml.SubElement(recipient_user, "thirty_day_stats", stats_attrs)

        return recipient_user

    def _sender_xml(self, root: xml.Element) -> xml.Element | None:
        sender_id = self.message.get("sender_user_id") if isinstance(self.message, dict) else None
        if not sender_id or pd.isna(sender_id) or str(sender_id).strip() == "":
            return None

        attrs = {"id": str(sender_id).strip(), "type": "user"}
        sender_elem = xml.SubElement(root, "sender_metadata", attrs)
        sender_elem.append(xml.Comment(" User or Business Sender Profile "))

        # Micro Interaction Stats calculated from message_events
        if self.message_events is not None and not self.message_events.empty:
            micro_attrs = {}
            if "message_opened" in self.message_events.columns:
                open_rate = round(self.message_events["message_opened"].mean(), 3)
                micro_attrs["open_rate"] = str(open_rate)

            if "message_replied" in self.message_events.columns:
                reply_rate = round(self.message_events["message_replied"].mean(), 3)
                micro_attrs["reply_rate"] = str(reply_rate)

            if micro_attrs:
                xml.SubElement(sender_elem, "micro_interaction_stats", micro_attrs)

        return sender_elem

    def _group_xml(self, root: xml.Element, groups: pd.DataFrame) -> xml.Element | None:
        group_id = self.message.get("group_id") if isinstance(self.message, dict) else None
        if not group_id or pd.isna(group_id) or str(group_id).strip() == "":
            return None

        group_row = groups.iloc[0] if (groups is not None and not groups.empty) else None
        member_row = self.group_members.iloc[0] if (self.group_members is not None and not self.group_members.empty) else None

        attrs = {"id": str(group_id).strip()}

        if group_row is not None and "group_type" in group_row and pd.notna(group_row["group_type"]) and str(group_row["group_type"]).strip() != "":
            attrs["group_type"] = str(group_row["group_type"]).strip()

        if member_row is not None:
            role_val = member_row.get("user_role", member_row.get("role"))
            if pd.notna(role_val) and str(role_val).strip() != "":
                attrs["user_role"] = str(role_val).strip()

            mute_val = member_row.get("user_mute_status", member_row.get("mute_state", member_row.get("group_muted_by_user")))
            if pd.notna(mute_val) and str(mute_val).strip() != "":
                if str(mute_val) in ["1", "1.0", "True", "Muted"]:
                    attrs["user_mute_status"] = "Muted"
                elif str(mute_val) in ["0", "0.0", "False", "Unmuted"]:
                    attrs["user_mute_status"] = "Unmuted"
                else:
                    attrs["user_mute_status"] = str(mute_val).strip()

            joined_val = member_row.get("joined_at", member_row.get("group_joined_at"))
            if pd.notna(joined_val) and str(joined_val).strip() != "":
                attrs["joined_at"] = str(joined_val).strip()

        group_elem = xml.SubElement(root, "group_metadata", attrs)
        group_elem.append(xml.Comment(" Basic information about each group chat along with how each user relates to each group "))

        if group_row is not None and "group_name" in group_row and pd.notna(group_row["group_name"]) and str(group_row["group_name"]).strip() != "":
            name_elem = xml.SubElement(group_elem, "group_name")
            name_elem.text = str(group_row["group_name"]).strip()

        return group_elem

    def _business_xml(self, root: xml.Element) -> xml.Element | None:
        business_id = self.message.get("business_id") if isinstance(self.message, dict) else None
        if not business_id or pd.isna(business_id) or str(business_id).strip() == "":
            return None

        business_row = self.business_accounts.iloc[0] if (self.business_accounts is not None and not self.business_accounts.empty) else None
        user_relationship_row = self.user_business_history.iloc[0] if (self.user_business_history is not None and not self.user_business_history.empty) else None

        attrs = {"id": str(business_id).strip()}

        if business_row is not None:
            field_mappings = [
                ("display_name", "name"), 
                ("business_name", "name"), 
                ("verification_tier", "verification_tier"), 
                ("category", "category"),
                ("account_age_days", "account_age_days"), 
                ("reports_count", "reports_count")
            ]

            for col_name, attr_name in field_mappings:
                if col_name in business_row and pd.notna(business_row[col_name]) and str(business_row[col_name]).strip() != "":
                    if attr_name not in attrs:  # Avoid overwriting
                        attrs[attr_name] = str(business_row[col_name]).strip()

        business_elem = xml.SubElement(root, "business_metadata", attrs)
        business_elem.append(xml.Comment(" Information about business senders "))

        if user_relationship_row is not None:
            user_relationship_attrs = {}
            user_relationship_mappings = [
                ("order_count", "order_count"), 
                ("last_transaction", "last_transaction"), 
                ("opt_in_status", "opt_in_status"),
                ("why_user_knows_account", "relation")
            ]
            for col_name, attr_name in user_relationship_mappings:
                if col_name in user_relationship_row and pd.notna(user_relationship_row[col_name]) and str(user_relationship_row[col_name]).strip() != "":
                    user_relationship_attrs[attr_name] = str(user_relationship_row[col_name]).strip()

            if user_relationship_attrs:
                xml.SubElement(business_elem, "user_relationship", user_relationship_attrs)

        return business_elem

    def _user_behavior_profile_xml(self, root: xml.Element) -> xml.Element | None:
        recipient_user_id = self.message.get("user_id") if isinstance(self.message, dict) else ""
        
        if self.message_history is None or self.message_history.empty:
            return None

        profile_elem = xml.SubElement(root, "user_behavior_profile", {"user_id": str(recipient_user_id).strip()})
        profile_elem.append(xml.Comment(" How users reacted to those past messages or has a recent relationship with a business "))

        # Merge message history with events if available
        if self.message_events is not None and not self.message_events.empty:
            merged = pd.merge(self.message_history.copy(), self.message_events, on=["message_id", "user_id"], how="left")
        else:
            merged = self.message_history.copy()

        # Limit to configured maximum historical messages (e.g. 5)
        history_subset = merged.tail(MAX_HISTORICAL_MESSAGES)
        
        interactions_elem = xml.SubElement(
            profile_elem, 
            "historical_interactions", 
            {"total_records": str(len(history_subset))}
        )

        for _, row in history_subset.iterrows():
            int_attrs = {"message_id": str(row["message_id"]).strip()}
            
            created_at = row.get("created_at")
            if pd.notna(created_at) and str(created_at).strip() != "":
                int_attrs["created_at"] = str(created_at).strip()

            sender_id = row.get("sender_user_id") if pd.notna(row.get("sender_user_id")) else row.get("business_id")
            if pd.notna(sender_id) and str(sender_id).strip() != "":
                int_attrs["sender_id"] = str(sender_id).strip()

            interaction = xml.SubElement(interactions_elem, "interaction", int_attrs)
            
            if "message_text" in row and pd.notna(row["message_text"]) and str(row["message_text"]).strip() != "":
                preview = xml.SubElement(interaction, "content_preview")
                preview.text = str(row["message_text"]).strip()

            behavior_attrs = {}
            behavior_cols = [
                ("message_opened", "opened"), 
                ("message_replied", "replied"), 
                ("behavior_time_minutes", "behavior_time_minutes"), 
                ("notification_dismissed", "dismissed"), 
                ("muted_after_message", "muted_after"), 
                ("message_reported", "reported")
            ]
            for col_name, attr_name in behavior_cols:
                if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip() != "":
                    behavior_attrs[attr_name] = str(row[col_name]).strip()

            if behavior_attrs:
                xml.SubElement(interaction, "behavior", behavior_attrs)

        return profile_elem

    def _media_attachment_xml(self, root: xml.Element) -> xml.Element | None:
        has_filepath = self.media_filepath and str(self.media_filepath).strip() != ""
        has_description = self.media_description and str(self.media_description).strip() != ""

        if not has_filepath and not has_description:
            return None

        media_elem = xml.SubElement(root, "media_attachment", {"status": "present"})

        media_id = self.message.get("media_id")
        media_type = self.message.get("media_type")

        comment = ""
        if media_type == "voice":
            comment = "- Voice Note Audio Transcription via Whisper"
        elif media_type == "image":
            comment = "- Extracted Image Text Optical Character Recognition (OCR) via Tesseract-OCR Engine"

        media_elem.append(xml.Comment(f" (Optional) Multimodal Media Attachment ({comment})" ))

        meta_attrs = {}
        if isinstance(self.message, dict):
            if media_id and pd.notna(media_id):
                meta_attrs["id"] = str(media_id).strip()
            if media_type and pd.notna(media_type):
                meta_attrs["type"] = str(media_type).strip()

        if has_filepath:
            meta_attrs["file_path"] = str(self.media_filepath).strip()

        if meta_attrs:
            xml.SubElement(media_elem, "metadata", meta_attrs)

        if has_description:
            desc_elem = xml.SubElement(media_elem, "description")
            desc_elem.text = str(self.media_description).strip()

        return media_elem
