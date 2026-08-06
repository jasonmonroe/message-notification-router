# src/data_handler.py

# +-----------------------------------+
# |           DATA HANDLER            |
# +-----------------------------------+

# Python Libraries
import csv
import cv2
import librosa as lr
import os

# Vendor Libraries
import pandas as pd
import numpy as np
from tinytag import TinyTag, ParseError

# Local Libraries
from src.constants import (
    AUDIO_SAMPLE_RATE, 
    CSV_FILENAMES,
    CSV_HEADER_COLS,
    DATASET_DIR, 
    MSEC,
    OUTPUT_FILEPATH,
)
from src.utils import show_banner

class DataHandler:
    def __init__(self, args: dict) -> None:
        self._use_sample = args.get("sample")
        
        # Text 
        self.business_accounts = None
        self.daily_notification_summary = None
        self.group_members = None
        self.groups = None
        self.images = None
        self.message_events = None
        self.message_history = None
        self.output = None
        self.user_business_history = None
        self.users = None
        self.voice_notes = None
        self.audios = None

        self._load()

    def _load(self) -> None:
        for csv_name in CSV_FILENAMES:
            filepath = os.path.join(DATASET_DIR, f"{csv_name}.csv")
            setattr(self, csv_name, pd.read_csv(filepath))

        # Override messages with sample if necessary
        if self._use_sample:
            self.messages = pd.read_csv(os.path.join(DATASET_DIR, "sample_messages.csv"))

    def save_output(self, csv_rows: list) -> None: 
        """
        Updates the existing CSV file by matching message_id using a list of lists.
        Preserves unprocessed placeholder rows and updates existing ones.
        """

        existing_data = {}
        
        # Read existing file if it already exists
        if os.path.exists(OUTPUT_FILEPATH):
            with open(OUTPUT_FILEPATH, "r", newline="", encoding="utf-8") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    msg_id = row.get("message_id")
                    if msg_id:  # Ignore empty or malformed rows
                        existing_data[msg_id] = row

        # Map incoming list of lists to the header columns and update/upsert
        for row_list in csv_rows:
            row_dict = dict(zip(CSV_HEADER_COLS, row_list))
            msg_id = row_dict.get("message_id")
            if msg_id:
                existing_data[msg_id] = row_dict

        # Write all rows back out cleanly
        with open(OUTPUT_FILEPATH, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_HEADER_COLS)
            writer.writeheader()
            for msg_id, row_data in existing_data.items():
                writer.writerow(row_data)

    def describe_audio(self) -> None:
        if self.voice_notes.empty:
            print("\n⚠️ Warning: No audio files loaded...")
            return None
        
        subtitles = [f"Audio File Count: {len(self.voice_notes)}"]
        show_banner(f"🔈 Describe Audio Data 🔈", subtitles)
    
        for row in self.voice_notes.itertuples():
            subtitles = []
            voice_note_id = row.voice_note_id
            file_path = row.file_path
            
            voice_note_filepath = os.path.join(DATASET_DIR, file_path)
            if not os.path.isfile(voice_note_filepath):
                print(f"\n⚠️ {voice_note_filepath} not a file.")
                continue
            
            subtitles.append(f"ℹ️ Filename: {voice_note_id}")
            subtitles.append(f"ℹ️ Filepath: {voice_note_filepath}")

            # Convert to audio file to audio data
            voice_note_data, sr = lr.load(voice_note_filepath, sr=AUDIO_SAMPLE_RATE)
            voice_note_duration = lr.get_duration(y=voice_note_data, sr=sr)
            subtitles.append(f"ℹ️ Duration: {voice_note_duration} seconds")
            subtitles.append(f"ℹ️ Shape: {voice_note_data.shape}")

            # Audio Metadata
            try:
                voice_note_tag = TinyTag.get(voice_note_filepath)
            except (ParseError, Exception) as e:

                # Fallback if audio file is currupted and can not be analyzed.
                print(f"\n⚠️ Warning: Could not parse audio file {voice_note_filepath}: {e}")
                continue

            subtitles.append("--- Basic Properties ---")
            subtitles.append(f"ℹ️ Filesize: {voice_note_tag.filesize} bytes")
            subtitles.append(f"ℹ️ Audio Offset: {voice_note_tag.audio_offset}")
            subtitles.append(f"ℹ️ Bitrate: {voice_note_tag.bitrate} kbps")
            subtitles.append(f"ℹ️ Sample Rate: {voice_note_tag.samplerate} Hz")
            subtitles.append(f"ℹ️ Channels: {voice_note_tag.channels}")
            subtitles.append(f"ℹ️ Bit Depth: {voice_note_tag.bitdepth}")

            subtitles.append("\n--- ID3 / Tag Metadata ---")
            subtitles.append(f"ℹ️ Title: {voice_note_tag.title}")
            subtitles.append(f"ℹ️ Artist: {voice_note_tag.artist}")
            subtitles.append(f"ℹ️ Album Artist: {voice_note_tag.albumartist}")
            subtitles.append(f"ℹ️ Album: {voice_note_tag.album}")
            subtitles.append(f"ℹ️ Composer: {voice_note_tag.composer}")
            subtitles.append(f"ℹ️ Track Number: {voice_note_tag.track}")
            subtitles.append(f"ℹ️ Track Total: {voice_note_tag.track_total}")
            subtitles.append(f"ℹ️ Disc Number: {voice_note_tag.disc}")
            subtitles.append(f"ℹ️ Disc Total: {voice_note_tag.disc_total}")
            subtitles.append(f"ℹ️ Year / Date: {voice_note_tag.year}")
            subtitles.append(f"ℹ️ Genre: {voice_note_tag.genre}")
            subtitles.append(f"ℹ️ Comment: {voice_note_tag.comment}")

            # Additional helper attributes TinyTag provides:
            subtitles.append("\n--- Additional Info ---")
            subtitles.append(f"ℹ️ File Format: {os.path.splitext(voice_note_filepath)[1]}")
            subtitles.append(f"ℹ️ Is Lossless?: {voice_note_tag.is_lossless}")
        
            show_banner(f"🔈 Audio File {row.Index}", subtitles)

    def describe_images(self) -> None:
        if self.images.empty:
            print("\n⚠️ Warning: No images loaded.")
            return None

        subtitles = [f"Image File Count: {len(self.images)}"]
        show_banner(f"🏞️ Describe Images 🏞️", subtitles)

        for row in self.images.itertuples():
            subtitles = []
            image_id = row.image_id
            file_path = row.file_path
            image_filepath = os.path.join(DATASET_DIR, file_path)

            if not os.path.isfile(image_filepath):
                print(f"\n⚠️ {image_filepath} is not a file.")
                continue

            subtitles.append(f"ℹ️ Filename: {image_id}")
            subtitles.append(f"ℹ️ Filesize: {os.path.getsize(image_filepath)}")

            # Convert to image data
            image_data = cv2.imread(image_filepath)

            if image_data is None:
                subtitles.append(f"\n⚠️ Warning: Failed to load image at {image_filepath}.")
                continue

            height, width, channels = image_data.shape
            subtitles.append(f"ℹ️ Height: {height}, Width: {width}")
            subtitles.append(f"ℹ️ Channels: {channels}")
            subtitles.append(f"ℹ️ Total Pixels:{image_data.size}") # Total number of elements
            subtitles.append(f"ℹ️ Data Type:{image_data.dtype}") # Usually uint8 (8-bit pixels)

            show_banner(f"🏞️ Image {row.Index}", subtitles)

            cv2.imshow(f"{row.Index} - {image_id}", image_data)

            # Delay x miliseconds before closing image
            cv2.waitKey(MSEC)
            cv2.destroyAllWindows()
        
    def describe(self) -> None:
        sample_text = "Sample" if self._use_sample else ""

        if self.messages.empty:
            raise ValueError(f"\n🚨 No {sample_text} messages loaded...")
        
        print("\n# --- 🗒️ Describe Text Data 🗒️ --- #")
        print(f"💬 {sample_text} Message Information")
        print(f"ℹ️ Row Count: {self.messages.shape[0]} | Column Count: {self.messages.shape[1]}")

        print("ℹ️ Head")
        print(self.messages.head())

        print("ℹ️ Tail")
        print(self.messages.tail())

        print("ℹ️ Shape")
        print(self.messages.shape)

        # Checking the data types of the columns for the dataset
        print("\nChecking the data types of the columns for the dataset:")
        print(self.messages.info())

        # Check for a Statistical Summary
        print("\nCheck for a Statistical Summary:")
        print(self.messages.describe().T)

        # Checking for missing values
        print("\nChecking for missing values:")
        print(self.messages.isnull().sum())

        # Checking for duplicate values
        num_dup_values = self.messages.duplicated().sum()
        print(f"Number of duplicate values: {num_dup_values}")

        self.describe_audio()
        self.describe_images()
