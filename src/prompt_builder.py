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
        #self.prompt = self.build(dataset.get("groups"))

    """

   

    <routing_input>
         <users>
            <user id="u_011" role="sender"/>
            <user id="u_006" role="recipient"/>
        </users>

        <!-- 1. LIVE INCOMING MESSAGE (from dataset/messages.csv) -->
        <incoming_message 
            id="msg_1042" 
            conversation_type="group" 
            group_id="group_003" 
            business_id="" 
            sender_id="u_011" 
            target_user_id="u_006" 
            created_at="2026-08-10 14:41:00" 
            media_type="image" 
            media_id="img_011" 
            forwarded_count="0">
            <message_text>School circular attached. Please check the timing and consent note.</message_text>
        </incoming_message>

        <!-- 2. RECIPIENT USER PROFILE & MACRO FATIGUE (users.csv + daily_notification_summary.csv) -->
        <recipient_user id="u_006" quiet_hours="21:00-06:00">
            <!-- Macro Fatigue from daily_notification_summary.csv -->
            <macro_notification_summary>
                <total_notifications_sent>82</total_notifications_sent>
                <total_notifications_dismissed>30</total_notifications_dismissed>
                <overall_dismissal_rate>0.366</overall_dismissal_rate>
            </macro_notification_summary>
            <!-- 30-Day Activity Baseline from users.csv -->
            <thirty_day_stats opened="80" replied="23" dismissed="2" reported="0" />
        </recipient_user>

        <!-- 3. SENDER SENSITIVITY & MICRO METRICS (from message_events.csv aggregated by sender) -->
        <sender_metadata id="u_011" type="user">
            <micro_interaction_stats open_rate="0.706" reply_rate="0.647" />
        </sender_metadata>

        <!-- 4. GROUP CHAT METADATA & RELATIONSHIP (groups.csv + group_members.csv) -->
        <group_metadata 
            id="group_003" 
            group_type="school_group" 
            user_role="admin" 
            user_mute_status="Unmuted" 
            joined_at="2024-06-15">
            <group_name>International School Parents Group</group_name>
        </group_metadata>

        <!-- 5. BUSINESS ACCOUNT METADATA & HISTORY (business_accounts.csv + user_business_history.csv) -->
        <!-- Included if conversation_type="business" or business_id is populated -->
        <business_metadata 
            id="b_001" 
            name="Acme Logistics" 
            verification_tier="verified" 
            account_age_days="450" 
            reports_count="0">
            <user_relationship 
                order_count="5" 
                last_transaction="2026-08-01" 
                opt_in_status="true" />
        </business_metadata>

        <!-- 6. USER BEHAVIOR PROFILE & HISTORICAL INTERACTIONS (message_history.csv + message_events.csv) -->
        <user_behavior_profile user_id="u_006">
            <historical_interactions total_records="2">
                <interaction message_id="message_0051" date="2026-06-11 07:08:00" sender_id="u_011">
                    <content_preview>Parents, the field-trip circular is attached. Please check pickup timing, consent form, and ID-card note before 6 PM.</content_preview>
                    <reaction opened="1" replied="1" reaction_time_minutes="2.0" dismissed="0" muted_after="0" reported="0" />
                </interaction>
                <interaction message_id="message_0182" date="2026-05-23 17:57:00" sender_id="u_011">
                    <content_preview>Route B bus is leaving 15 minutes early today because the main signal road is blocked.</content_preview>
                    <reaction opened="1" replied="0" reaction_time_minutes="12.0" dismissed="0" muted_after="0" reported="0" />
                </interaction>
            </historical_interactions>
        </user_behavior_profile>

        <!-- 7. MULTIMODAL MEDIA ATTACHMENTS (images.csv / voice_notes.csv + OCR / ASR) -->
        <media_attachment status="present">
            <metadata id="img_011" type="image" file_path="media/images/img_011.jpg" />
            <description><![CDATA[Extracted Image Text (OCR): FIELD TRIP CONSENT FORM
    Trip Destination: City Zoo | Date: 2026-08-15
    Departure Time: 08:00 AM | Return Time: 03:00 PM
    Transportation: School Bus
    Signature of Parent/Guardian required for participation.]]></description>
        </media_attachment>
    </routing_input>
    """

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
        target_user_id = self.message.get("user_id") if isinstance(self.message, dict) else None

        valid_sender = pd.notna(sender_id) and str(sender_id).strip() != ""
        valid_recipient = pd.notna(target_user_id) and str(target_user_id).strip() != ""

        if not valid_sender and not valid_recipient:
            return None

        users = xml.SubElement(root, "users")

        if valid_sender:
            xml.SubElement(users, "user", id=str(sender_id).strip(), role="sender")

        if valid_recipient:
            xml.SubElement(users, "user", id=str(target_user_id).strip(), role="recipient")

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
            "target_user_id": msg.get("user_id"),
            "created_at": msg.get("created_at"),
            "media_type": msg.get("media_type"),
            "media_id": msg.get("media_id"),
            "forwarded_count": msg.get("forwarded_count"),
        }

        for attr_name, val in field_map.items():
            if pd.notna(val) and str(val).strip() != "":
                attrs[attr_name] = str(val).strip()

        incoming_message = xml.SubElement(root, "incoming_message", attrs)

        text_val = msg.get("message_text")
        if pd.notna(text_val) and str(text_val).strip() != "":
            xml_text = xml.SubElement(incoming_message, "message_text")
            xml_text.text = str(text_val).strip()

        return incoming_message

    def _recipient_user_xml(self, root: xml.Element) -> xml.Element | None:
        target_user_id = self.message.get("user_id") if isinstance(self.message, dict) else None
        
        user_row = None
        if self.users is not None and not self.users.empty:
            user_row = self.users.iloc[0]

        attrs = {}
        if pd.notna(target_user_id) and str(target_user_id).strip() != "":
            attrs["id"] = str(target_user_id).strip()

        if user_row is not None:
            dnd = user_row.get("do_not_disturb_window") if "do_not_disturb_window" in user_row else user_row.get("dnd_window")
            if pd.notna(dnd) and str(dnd).strip() != "":
                attrs["do_not_disturb_window"] = str(dnd).strip()

        recipient_user = xml.SubElement(root, "recipient_user", attrs)

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


    def _recipient_user_xml_old(self, root: xml.Element) -> xml.Element | None:
        target_user_id = self.message.get("user_id") if isinstance(self.message, dict) else None
        
        user_row = None
        if self.users is not None and not self.users.empty:
            user_row = self.users.iloc[0]

        attrs = {}
        if pd.notna(target_user_id) and str(target_user_id).strip() != "":
            attrs["id"] = str(target_user_id).strip()

        if user_row is not None:
            dnd = user_row.get("do_not_disturb_window") if "do_not_disturb_window" in user_row else user_row.get("dnd_window")
            if pd.notna(dnd) and str(dnd).strip() != "":
                attrs["do_not_disturb_window"] = str(dnd).strip()

        recipient_user = xml.SubElement(root, "recipient_user", attrs)

        # Macro Notification Summary from daily_notification_summary dataframe
        if self.daily_notification_summary is not None and not self.daily_notification_summary.empty:
            sent_sum = self.daily_notification_summary["notifications_sent"].sum() if "notifications_sent" in self.daily_notification_summary else 0
            dismissed_sum = self.daily_notification_summary["notifications_dismissed"].sum() if "notifications_dismissed" in self.daily_notification_summary else 0

            if sent_sum > 0 or dismissed_sum > 0:
                macro_elem = xml.SubElement(recipient_user, "macro_notification_summary")
                xml.SubElement(macro_elem, "total_notifications_sent").text = str(int(sent_sum))
                xml.SubElement(macro_elem, "total_notifications_dismissed").text = str(int(dismissed_sum))
                
                rate = round(dismissed_sum / max(sent_sum, 1), 3)
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

        if group_row is not None and "group_name" in group_row and pd.notna(group_row["group_name"]) and str(group_row["group_name"]).strip() != "":
            name_elem = xml.SubElement(group_elem, "group_name")
            name_elem.text = str(group_row["group_name"]).strip()

        return group_elem

    def _business_xml(self, root: xml.Element) -> xml.Element | None:
        bus_id = self.message.get("business_id") if isinstance(self.message, dict) else None
        if not bus_id or pd.isna(bus_id) or str(bus_id).strip() == "":
            return None

        bus_row = self.business_accounts.iloc[0] if (self.business_accounts is not None and not self.business_accounts.empty) else None
        rel_row = self.user_business_history.iloc[0] if (self.user_business_history is not None and not self.user_business_history.empty) else None

        attrs = {"id": str(bus_id).strip()}

        if bus_row is not None:
            field_mappings = [
                ("display_name", "name"), 
                ("business_name", "name"), 
                ("verification_tier", "verification_tier"), 
                ("category", "category"),
                ("account_age_days", "account_age_days"), 
                ("reports_count", "reports_count")
            ]

            for col_name, attr_name in field_mappings:
                if col_name in bus_row and pd.notna(bus_row[col_name]) and str(bus_row[col_name]).strip() != "":
                    if attr_name not in attrs:  # Avoid overwriting
                        attrs[attr_name] = str(bus_row[col_name]).strip()

        bus_elem = xml.SubElement(root, "business_metadata", attrs)

        if rel_row is not None:
            rel_attrs = {}
            rel_mappings = [
                ("order_count", "order_count"), 
                ("last_transaction", "last_transaction"), 
                ("opt_in_status", "opt_in_status"),
                ("why_user_knows_account", "relation")
            ]
            for col_name, attr_name in rel_mappings:
                if col_name in rel_row and pd.notna(rel_row[col_name]) and str(rel_row[col_name]).strip() != "":
                    rel_attrs[attr_name] = str(rel_row[col_name]).strip()

            if rel_attrs:
                xml.SubElement(bus_elem, "user_relationship", rel_attrs)

        return bus_elem

    def _user_behavior_profile_xml(self, root: xml.Element) -> xml.Element | None:
        target_user_id = self.message.get("user_id") if isinstance(self.message, dict) else ""
        
        if self.message_history is None or self.message_history.empty:
            return None

        profile_elem = xml.SubElement(root, "user_behavior_profile", {"user_id": str(target_user_id).strip()})
        profile_elem.append(xml.Comment("USER BEHAVIOR PROFILE & HISTORICAL INTERACTIONS"))

        # Merge message history with events if available
        if self.message_events is not None and not self.message_events.empty:
            merged = pd.merge(self.message_history, self.message_events, on=["message_id", "user_id"], how="left")
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
            
            created_at = row.get("created_at", row.get("date"))
            if pd.notna(created_at) and str(created_at).strip() != "":
                int_attrs["date"] = str(created_at).strip()

            sender_id = row.get("sender_user_id") if pd.notna(row.get("sender_user_id")) else row.get("business_id")
            if pd.notna(sender_id) and str(sender_id).strip() != "":
                int_attrs["sender_id"] = str(sender_id).strip()

            interaction = xml.SubElement(interactions_elem, "interaction", int_attrs)

            if "message_text" in row and pd.notna(row["message_text"]) and str(row["message_text"]).strip() != "":
                preview = xml.SubElement(interaction, "content_preview")
                preview.text = str(row["message_text"]).strip()

            rx_attrs = {}
            reaction_cols = [
                ("message_opened", "opened"), 
                ("message_replied", "replied"), 
                ("reaction_time_minutes", "reaction_time_minutes"), 
                ("notification_dismissed", "dismissed"), 
                ("muted_after_message", "muted_after"), 
                ("message_reported", "reported")
            ]
            for col_name, attr_name in reaction_cols:
                if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip() != "":
                    rx_attrs[attr_name] = str(row[col_name]).strip()

            if rx_attrs:
                xml.SubElement(interaction, "reaction", rx_attrs)

        return profile_elem

    def _media_attachment_xml(self, root: xml.Element) -> xml.Element | None:
        has_filepath = self.media_filepath and str(self.media_filepath).strip() != ""
        has_description = self.media_description and str(self.media_description).strip() != ""

        if not has_filepath and not has_description:
            return None

        media_elem = xml.SubElement(root, "media_attachment", {"status": "present"})
        media_elem.append(xml.Comment("(Optional) MULTIMODAL MEDIA ATTACHMENTS (Voice Note Audio Transcription via Whisper)"))

        meta_attrs = {}
        if isinstance(self.message, dict):
            if self.message.get("media_id") and pd.notna(self.message.get("media_id")):
                meta_attrs["id"] = str(self.message.get("media_id")).strip()
            if self.message.get("media_type") and pd.notna(self.message.get("media_type")):
                meta_attrs["type"] = str(self.message.get("media_type")).strip()

        if has_filepath:
            meta_attrs["file_path"] = str(self.media_filepath).strip()

        if meta_attrs:
            xml.SubElement(media_elem, "metadata", meta_attrs)

        if has_description:
            desc_elem = xml.SubElement(media_elem, "description")
            desc_elem.text = str(self.media_description).strip()

        return media_elem


    # --- Format Attributes  (old way) --- #

     # @todo - old way
    def build_orig(self, groups: pd.DataFrame) -> str:
        return ROUTING_PROMPT_TEMPLATE.format(            
            business_sender_context=self._format_business_sender_xml(),
            group_metadata_context=self._format_group_metadata_xml(groups),
            historical_evidence=self._format_historical_evidence_xml(),
            incoming_message_context=self._format_incoming_message_xml(),
            media_context=self._format_media_context_xml(),
            
            message_id=self.message.get("message_id"),
            recipient_user_context=self._format_recipient_user_context_xml(),
            user_behavioral_profile_context=self._format_user_behavioral_profile_context_xml(),
        )

    def _format_incoming_message_xml(self) -> str:
        group_id = self.message.get("group_id") if pd.notna(self.message.get("group_id")) and self.message.get("group_id").strip() else ""
        sender_or_business_id = self.message.get("business_id") if pd.notna(self.message.get("business_id")) else (self.message.get("send_user_id") if pd.notna(self.message.get("send_user_id")) else "")
        message_text = self.message.get("message_text") if pd.notna(self.message.get("message_text")) else "[Media Message]"
        media_type = self.message.get("media_type") if pd.notna(self.message.get("media_type")) else "None"
         
        return """<incoming_message id="{message_id}" conversation_type="{conversation_type}" group_id="{group_id}" sender_id="{sender_or_business_id}" target_user_id="{user_id}" created_at="{created_at}" media_type="{media_type}" forwarded_count="{forwarded_count} ">\n\t<message_text>{message_text}</message_text>\n</incoming_message>""".strip().format(
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

                business_info = """<business_sender id="{business_id}" type="sender" name="{name}" category="{category}" verified="{verified}" sender_domain="{sender_domain}">""".strip().format(
                    business_id=self.message.get("business_id"),
                    name=business_row.get("display_name", self.message.get("brand_name")),
                    category=business_row.get("category"),
                    verified="true" if business_row.get("verified") == 1 else "false",
                    sender_domain=business_row.get("domain_used_by_sender", ""),
                )

            return business_info

        def _get_history_info_xml(business_history_df: pd.DataFrame) -> str:

            history_info_xml = ""
            if business_history_df is not None and not business_history_df.empty:

                # Filter history for this specific business/user if needed and dynamically check for whatever columns actually exist in your dataframe.
                history_rows = []
                for _, row in business_history_df.iterrows():
                    # Use real columns present in your dataset rows: action, message_type, created_at     
                    xml_content = """<interaction user_id="{user_id}" relation="{why_user_knows_account}" last_reply_at="{last_reply_at}" last_activity="{last_activity_at}"/>""".strip().format(
                        last_activity_at=row.get("last_activity_at") if row.get("last_activity_at") else "",
                        last_reply_at=row.get("last_reply_at") if pd.notna(row.get("last_reply_at")) and str(row.get("last_reply_at")).strip() else "",
                        user_id=row.get("user_id"),
                        why_user_knows_account=row.get("why_user_knows_account"),
                    )
                
                    # Format each row as a clean self-closing XML tag
                    history_rows.append(xml_content)
                    
                history_info_xml = "\n".join(history_rows)
           
            return history_info_xml

        business_info_xml_open = _get_business_info_xml_open(business_df)
        history_info_xml = _get_history_info_xml(business_history_df)
        business_info_xml_close = "</business_sender>\n"

        return f"""{business_info_xml_open}\n\t<interactions>\n\t\t{history_info_xml}\n\t</interactions>\n{business_info_xml_close}\n""".strip() 

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
        return """<group_metadata id="{group_id}" type="{group_type}" user_role="{group_user_role}" joined_at="{group_joined_at}" user_mute_status="{group_muted_by_user}">\n\t<group_name>{group_name}</group_name>\n</group_metadata>""".strip().format(
            group_id=self.message.get("group_id"),
            group_name=group_name,
            group_type=group_type,
            group_joined_at=group_members_row["joined_at"].iloc[0],
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
            message_text = row.get("message_text") if pd.notna(row.get("message_text")) and str(row.get("message_text")).strip() else ""
    
            xml_message = """<message id="{message_id}" date="{created_at}">{message_text}</message>""".strip().format(
                message_id=row.get("message_id"),
                created_at=row.get("created_at"),
                message_text=message_text.strip(),
            )

            xml_messages.append(f"\t{xml_message}")
       
        xml_content = "\n".join(xml_messages)
        
        return f"{xml_open}\n{xml_content}\n{xml_close}".strip()

    def _format_media_context_xml(self) -> str:
        if self.media_description:
            return """<media_attachment status="optional">\n\t<metadata id="{media_id}" type="{media_type}" filename="{media_filename}" />\n\t<description>{media_content_description}</description>\n</media_attachment>""".strip().format(
                    media_id=self.message.get("media_id"),
                    media_type=self.message.get("media_type"),
                    media_filename=os.path.basename(self.media_filepath),
                    media_content_description=self.media_description.strip(),
                )

        return ""

    def _format_recipient_user_context_xml(self) -> str:
        # No specific user profile data found.
        if self.users is None or self.users.empty:
            return ""
        
        # User row from users.csv
        user = self.users.iloc[0]

        return """<recipient_user id="{user_id}" type="recipient" dnd_window="{do_not_disturb_window}">\n\t<stats thirty_day_opened="{messages_opened_30d}" thirty_day_replied="{messages_replied_30d}" thirty_day_dismissed="{messages_reported_30d}" thirty_day_reported="{notifications_dismissed_30d}" />\n</recipient_user>""".strip().format(
            do_not_disturb_window=user.get("do_not_disturb_window", "None"),
            messages_opened_30d=user.get("messages_opened_30d", 0),
            messages_replied_30d=user.get("messages_replied_30d", 0),
            messages_reported_30d=user.get("messages_reported_30d", 0),
            notifications_dismissed_30d=user.get("notifications_dismissed_30d", 0),
            user_id=user.get("user_id", self.message.get("user_id")),
        )

    def _format_user_behavioral_profile_context_xml(self) -> str:
        return ""

        return """<user_behavioral_profile user_id="{user_id}">\n\t
        <></>\n\t
        <></>\n
        </user_behavioral_profile>""".strip().format(
            user_id=self.message.get("user_id"),
        )
