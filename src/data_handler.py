# src/data_handler.py

# Python Libraries
import csv
import json
import cv2
import librosa as lr
import os

# Vendor Libraries
import pandas as pd
import numpy as np
from tinytag import TinyTag

# Local Libraries
from src.constants import (
    AUDIO_DIR, 
    AUDIO_SAMPLE_RATE, 
    CSV_FILENAMES,
    CSV_HEADER_COLS,
    DATASET_DIR, 
    IMAGES_DIR,
    MSEC,
    OUTPUT_FILE,
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

        # Media
        #self.audios = self._load_audio()
        #self.images = self._load_images()

    def _load(self) -> None:
        print("Loading text files...")
        
        for csv_name in CSV_FILENAMES:
            filepath = os.path.join(DATASET_DIR, f"{csv_name}.csv")

            setattr(self, csv_name, pd.read_csv(filepath))

        # Override messages with sample if necessary
        if self._use_sample:
            self.messages = pd.read_csv(os.path.join(DATASET_DIR, "sample_messages.csv"))

    def _format_output(self, results: dict) -> pd.DataFrame | None:
        
        # Prepare the row values (serialize lists if needed)
        row_data = results.copy()
        if isinstance(row_data.get("evidence_message_ids"), list):
            row_data["evidence_message_ids"] = json.dumps(row_data["evidence_message_ids"])
        
        message_id = row_data.get("message_id")
        output_df = None
        
        # Find and update the matching row, or append if it doesn't exist
        if message_id in output_df["message_id"].values:
            for column, value in row_data.items():
                if column in output_df.columns:
                    output_df.loc[output_df["message_id"] == message_id, column] = value
        else:
            output_df = pd.concat([output_df, pd.DataFrame([row_data])], ignore_index=True)

        return output_df

    def save_output(self, csv_rows: list) -> None:
          
        # Write all at once
        with open("output.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER_COLS) 
            writer.writerows(csv_rows)           


    # @TODO - old version
    def save_output2(self, results: dict) -> None:
        print("\nsave_output()")

        # Read the file directly from disk to ensure it's up to date in loops
        output_filepath = os.path.join(DATASET_DIR, OUTPUT_FILE)
        output_df = pd.read_csv(output_filepath) if os.path.exists(output_filepath) else self.output.copy()

        formatted_output_df = self._format_output(results)
        if formatted_output_df is not None:
            output_df = formatted_output_df
        
        # Write the complete updated DataFrame back to the CSV file
        print(f"output_filepath={output_filepath}")
        print(f"output_df={output_df}")

        output_df.to_csv(output_filepath, index=False)

    def describe_audio(self) -> None:
        if not self.voice_notes:
            print("\n⚠️ Warning: No audio files loaded...")
            return None

        #print("\n# --- 🔈 Describe Audio Data 🔈 --- #")
        #print(f"ℹ️  Audio File Count: {len(self.voice_notes)}")

        subtitles = []
        for idx, voice_note_id, file_path in enumerate(self.voice_notes):

            if not os.path.isfile(file_path):
                continue

            subtitles.append("\n🔈 Audio File {idx}")
            subtitles.append(f"ℹ️Filename: {voice_note_id}")
            subtitles.append(f"ℹ️Filepath: {file_path}")

            # Convert to audio file to audio data
            audio_data, sr = lr.load(file_path, sr=AUDIO_SAMPLE_RATE)
            audio_duration = lr.get_duration(y=audio_data, sr=sr)
            subtitles.append(f"ℹ️Duration: {audio_duration} seconds")
            subtitles.append(f"ℹ️ Shape: {audio_data.shape}")

            # Audio Metadata
            audio_tag = TinyTag.get(file_path)

            subtitles.append("--- Basic Properties ---")
            subtitles.append(f"ℹ️Filesize: {audio_tag.filesize} bytes")
            subtitles.append(f"ℹ️Audio Offset: {audio_tag.audio_offset}")
            subtitles.append(f"ℹ️Bitrate: {audio_tag.bitrate} kbps")
            subtitles.append(f"ℹ️Sample Rate: {audio_tag.samplerate} Hz")
            subtitles.append(f"ℹ️Channels: {audio_tag.channels}")
            subtitles.append(f"ℹ️Bit Depth: {audio_tag.bitdepth}")

            subtitles.append("\n--- ID3 / Tag Metadata ---")
            subtitles.append(f"ℹ️Title: {audio_tag.title}")
            subtitles.append(f"ℹ️Artist: {audio_tag.artist}")
            subtitles.append(f"ℹ️Album Artist: {audio_tag.albumartist}")
            subtitles.append(f"ℹ️Album: {audio_tag.album}")
            subtitles.append(f"ℹ️Composer: {audio_tag.composer}")
            subtitles.append(f"ℹ️Track Number: {audio_tag.track}")
            subtitles.append(f"ℹ️Track Total: {audio_tag.track_total}")
            subtitles.append(f"ℹ️Disc Number: {audio_tag.disc}")
            subtitles.append(f"ℹ️Disc Total: {audio_tag.disc_total}")
            subtitles.append(f"ℹ️Year / Date: {audio_tag.year}")
            subtitles.append(f"ℹ️Genre: {audio_tag.genre}")
            subtitles.append(f"ℹ️Comment: {audio_tag.comment}")

            # Additional helper attributes TinyTag provides:
            subtitles.append("\n--- Additional Info ---")
            subtitles.append(f"ℹ️File Format: {audio_tag.file_format}")
            subtitles.append(f"ℹ️Is Lossless?: {audio_tag.is_lossless}")

        show_banner(f"🔈 Describe Audio Data ({len(self.voice_notes)}) 🔈", subtitles)

    def describe_images(self) -> None:
        if not self.images:
            print("\n⚠️ Warning: No images loaded.")
            return None

        subtitles = []
        for idx, image_id, file_path in enumerate(self.images):
            if not os.path.isfile(file_path):
                continue

            subtitles.append(f"🏞️ Image {idx}")
            subtitles.append(f"ℹ️ Filename: {image_id}")
            subtitles.append(f"ℹ️ Filesize: {os.path.getsize(image_id)}")

            # Convert to image data
            image_data = cv2.imread(file_path)

            if not image_data:
                subtitles.append(f"⚠️ Warning: Failed to load image at {file_path}.")
                continue

            height, width, channels = image_data.shape
            subtitles.append(f"ℹ️ Height: {height}, Width: {width}")
            subtitles.append(f"ℹ️ Channels: {channels}")
            subtitles.append(f"ℹ️ Total Pixels:{image_data.size}") # Total number of elements
            subtitles.append(f"ℹ️ Data Type:{image_data.dtype}") # Usually uint8 (8-bit pixels)

            cv2.imshow(image_id, image_data)

            # Delay x miliseconds before closing image
            cv2.waitKey(MSEC)
            cv2.destroyAllWindows()
        
        show_banner(f"🏞️ Describe Image Data ({len(self.images)}) 🏞️", subtitles)

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


    # @TODO - do I even need?
    def _get_audio(self) -> list:
        # https://lr.org/doc/latest/index.html
        # https://www.comet.com/site/blog/working-with-audio-data-for-machine-learning-in-python/
        
        print("Loading audio files...")

        if not os.path.exists(AUDIO_DIR):
            print(f"\n🚨 Error: Directory not found -> {AUDIO_DIR} 🚨.")
            return []

        audio_files = []
        for idx, audio_filename in enumerate(os.listdir(AUDIO_DIR)):
            audio_filepath = os.path.join(AUDIO_DIR, audio_filename)
            
            if not os.path.isfile(audio_filepath):
                continue
                
            print(f"\n🔈 Audio File {idx}")

            audio_data, sr = lr.load(audio_filepath, sr=AUDIO_SAMPLE_RATE)
            #print(f"ℹ️ Shape: {audio_data.shape}")

            # Calculate duration using the loaded array data
            duration = lr.get_duration(y=audio_data, sr=sr)
            #print(f"ℹ️ Duration: {duration:.2f} seconds")

            #print(f"ℹ️ Filename: {audio_filename}")
            
            # Load the raw waveform array and sampling rate
            audio_dataset = {
                "filename": audio_filename,
                "filepath": audio_filepath,
                "data": audio_data,
                "duration": duration,
                "shape": audio_data.shape
            }

        # Use TinyTag on the file path to extract metadata tags securely
        try:
            audio_tag = TinyTag.get(audio_filepath)
            
            # Add attributes to the dictionary
            audio_dataset["title"] = audio_tag.title or "Unknown"
            audio_dataset["artist"] = audio_tag.artist
            audio_dataset["albumartist"] = audio_tag.albumartist
            audio_dataset["album"] = audio_tag.album
            audio_dataset["composer"] = audio_tag.composer
            audio_dataset["track"] = audio_tag.track
            audio_dataset["track_total"] = audio_tag.track_total
            audio_dataset["disc"] = audio_tag.disc
            audio_dataset["disc_total"] = audio_tag.disc_total
            audio_dataset["year"] = audio_tag.year
            audio_dataset["genre"] = audio_tag.genre
            audio_dataset["comment"] = audio_tag.comment
            audio_dataset["filesize"] = audio_tag.filesize
            audio_dataset["audio_offset"] = audio_tag.audio_offset
            audio_dataset["bitrate"] = audio_tag.bitrate
            audio_dataset["samplerate"] = audio_tag.samplerate
            audio_dataset["channels"] = audio_tag.channels
            audio_dataset["bitdepth"] = audio_tag.bitdepth
            audio_dataset["file_format"] = audio_tag.file_format
            audio_dataset["is_lossless"] = audio_tag.is_lossless

            print("--- Basic Properties ---")
            print(f"Duration: {audio_tag.duration} seconds")
            print(f"Filesize: {audio_tag.filesize} bytes")
            print(f"Audio Offset: {audio_tag.audio_offset}")
            print(f"Bitrate: {audio_tag.bitrate} kbps")
            print(f"Sample Rate: {audio_tag.samplerate} Hz")
            print(f"Channels: {audio_tag.channels}")
            print(f"Bit Depth: {audio_tag.bitdepth}")
            
            print("\n--- ID3 / Tag Metadata ---")
            print(f"Title: {audio_tag.title}")
            print(f"Artist: {audio_tag.artist}")
            print(f"Album Artist: {audio_tag.albumartist}")
            print(f"Album: {audio_tag.album}")
            print(f"Composer: {audio_tag.composer}")
            print(f"Track Number: {audio_tag.track}")
            print(f"Track Total: {audio_tag.track_total}")
            print(f"Disc Number: {audio_tag.disc}")
            print(f"Disc Total: {audio_tag.disc_total}")
            print(f"Year / Date: {audio_tag.year}")
            print(f"Genre: {audio_tag.genre}")
            print(f"Comment: {audio_tag.comment}")
            
            # Additional helper attributes TinyTag provides:
            print(f"\n--- Additional Info ---")
            print(f"File Format: {audio_tag.file_format}")
            print(f"Is Lossless?: {audio_tag.is_lossless}")

        except Exception:
            print("ℹ️ Title: Could not read metadata.")

            audio_files.append(audio_dataset)
            
        return audio_files

    # @TODO - do I even need?
    def _get_images(self) -> list:
        print("Loading images...")

        if not os.path.exists(IMAGES_DIR):
            print(f"\n🚨 Error: Directory not found -> {IMAGES_DIR} 🚨.\n")
            return []

        images = []
        for image_filename in os.listdir(IMAGES_DIR):
            image_filepath = os.path.join(IMAGES_DIR, image_filename)

            image_dataset = {
                "filename": image_filename,
                "filepath": image_filepath,
            }
            
            # Skip directories if any accidentally exist inside the folder
            if not os.path.isfile(image_filepath):
                continue
            
            # Read the image
            img_data = cv2.imread(image_filepath)
            
            # Verify OpenCV actually read the file successfully
            if img_data is not None:
                image_dataset["data"] = img_data

                images.append(img_data)
            else:
                print(f"⚠️ Warning: Failed to load image at {image_filepath}")
                
        return images


 # @TODO old version
    def describe_audio2(self) -> None:
        if not self.audios:
            print("\n⚠️ Warning: No audio files loaded...")
            return
            
        print("\n# --- 🔈 Describe Audio Data 🔈 --- #")
        print(f"ℹ️  Audio File Count: {len(self.audios)}")
        
        audio_filenames = os.listdir(AUDIO_DIR)
        print(f"audio_files={audio_filenames}")
        
        for idx, audio in enumerate(self.audios):      
            print(f"\n🔈 Audio File {idx}")

            for key, value in audio.items():
                print(f"ℹ️ {key.title()}: {value}")
               
    # @todo old version
    def describe_images2(self) -> None:

        if not self.images:
            print("\n⚠️ Warning: No images loaded.")
            return None
      
        print("\n# --- 🏞️ Describe Image Data 🏞️ --- #")
        print(f"Image Count: {len(self.images)}")

        image_files = os.listdir(IMAGES_DIR)
        for idx, image in enumerate(self.images):
            image_filepath = image_files[idx]
            image_filename, _ = os.path.splitext(image_filepath)

            print(f"\n🏞️ Image {idx}")
            print(f"\tℹ️ Filename: {image_filename}")
            

            height, width, channels = image.shape
            print(f"\tℹ️ Height: {height}, Width: {width}, Channels: {channels}")



            total_pixels = img_data.size              # Total number of elements
            data_type = img_data.dtype                # Usually uint8 (8-bit pixels)

    
            cv2.imshow(image_filename, image)