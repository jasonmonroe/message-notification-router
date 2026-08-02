# src/data_handler.py

# Python Libraries
import cv2
import librosa as lr
import os
from pathlib import Path

# Vendor Libraries
import pandas as pd
import numpy as np
from tinytag import TinyTag

# Local Libraries
from src.constants import (
    AUDIO_DIR, 
    AUDIO_SAMPLE_RATE, 
    CSV_FILENAMES, 
    DATASET_DIR, 
    IMAGES_DIR,
)


class DataHandler:
    def __init__(self, args: dict) -> None:
        self._use_sample = args.get("sample")
        
        self.data = None
        self.df = None

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

        self._load_text()

        # Media
        self.audios = self._load_audio()
        self.images = self._load_images()

    def _load_text(self) -> None:
        print("Loading text files...")
        
        # Text
        filename = "sample" if self._use_sample else ""
        filename = f"{filename}_messages.csv"
        self.messages = pd.read_csv(os.path.join(DATASET_DIR, filename))

        # Supporting Text files (non-messages)
        for csv_name in CSV_FILENAMES:
            if csv_name in ["messages", "output"]:
                continue

            setattr(self, csv_name, pd.read_csv(os.path.join(DATASET_DIR, f"{csv_name}.csv")))
            

        #self.business_accounts = pd.read_csv(os.path.join(DATASET_DIR, "business_accounts.csv"))
        #self.daily_notification_summary = pd.read_csv(os.path.join(DATASET_DIR, "daily_notification_summary.csv"))
        #self.group_members = pd.read_csv(os.path.join(DATASET_DIR, "group_members.csv"))
        #self.groups = pd.read_csv(os.path.join(DATASET_DIR, "groups.csv"))
        #self.images = pd.read_csv(os.path.join(DATASET_DIR, "images.csv"))
        #self.message_events = pd.read_csv(os.path.join(DATASET_DIR, "message_events.csv"))
        #self.message_history = pd.read_csv(os.path.join(DATASET_DIR, "message_history.csv"))
        #self.user_business_history = pd.read_csv(os.path.join(DATASET_DIR, "user_business_history.csv"))
        #self.users = pd.read_csv(os.path.join(DATASET_DIR, "users.csv"))
        #self.voice_notes = pd.read_csv(os.path.join(DATASET_DIR, "voice_notes.csv"))


    def get_csv_file(self, filename: str):
        if filename:
            filepath = f"{DATASET_DIR}{filename}.csv"
            return pd.read_csv(filepath)
        

    def _load_audio(self) -> list:
        # https://lr.org/doc/latest/index.html
        # https://www.comet.com/site/blog/working-with-audio-data-for-machine-learning-in-python/

        print("Loading audio files...")

        #import sys

        #print(f"f={AUDIO_DIR }")
        #sys.exit(0)
        
        audio_files = []
        for audio_filename in os.listdir(AUDIO_DIR):
            print(audio_filename)
            sys.exit(0)
            audio_filepath = os.path.join(AUDIO_DIR, audio_filename)
            print(f"audio_filename = {audio_filename}, audio_filepath={audio_filepath}")
            audio_file = lr.load(audio_filepath, sr=AUDIO_SAMPLE_RATE)
            audio_files.append(audio_file)
        print("line 93")
        return audio_files

    def _load_images(self) -> list:
        print("Loading images...")

        images = []
        for image_path in os.listdir(IMAGES_DIR):
            img_array = cv2.imread(image_path) 
            images.append(img_array)

        return images

    def describe(self) -> None:
        sample_text = "Sample" if self._use_sample else ""

        if self.messages.empty:
            ValueError(f"\n🚨 No {sample_text} messages loaded...")
            return None
        
        print("\n# --- 🗒️ Describe Text Data 🗒️ --- #")
        print(f"💬 {sample_text} Message Information")
        print(f"ℹ️ Column Count: {self.messages.shape[0]} | Row Count: {self.messages.shape[1]}")
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
        
             
    def describe_audio(self) -> None:
        # Media Content

        if not self.audios:
            print("\n⚠️ Warning: No audio files loaded...")
            return
        
        print("# --- 🔈 Describe Audio Data 🔈 --- #")
        print(f"ℹ️ Audio File Count: {len(self.audios)}")

        audio_files = os.listdir(AUDIO_DIR)
        for idx, audio in enumerate(self.audios):
            audio_filepath = audio_files[idx]
            audio_filename = audio_filename.replace(f"{AUDIO_DIR}", "")
            audio_tag = TinyTag.get(audio_filepath)
            
            print(f"🔈 Audio File {idx}")
            print(f"\tℹ️ Shape: {audio.shape}")
            print(f"\tℹ️ Sample Rate: {lr.get_samplerate(audio)}")
            print(f"\tℹ️ Duration: {lr.get_duration(audio):.2f} seconds")
            
            print(f"\tℹ️ Title: {audio_tag.title}")
            print(f"\tℹ️ Artist: {audio_tag.artist}")
            print(f"\tℹ️ Album: {audio_tag.album}")
            print(f"\tℹ️ Bitrate: {audio_tag.bitrate} kbps")
        
            

    def describe_images(self) -> None:

        if not self.images:
            print("\n⚠️ Warning: No images loaded.")
            return None
      
        print("# --- 🏞️ Describe Image Data 🏞️ --- #")
        print(f"Image Count: len{self.images}")

        image_files = os.listdir(IMAGES_DIR)
        for idx, image in enumerate(self.images):
            image_filepath = image_files[idx]
            image_filename = image_filepath.replace(f"{IMAGES_DIR}", "")

            print(f"🏞️ Image {idx}")
            print(f"\tℹ️ Filename: {image_filename}")

            height, width, channels = image.shape
            print(f"\tℹ️ Height: {height}, Width: {width}, Channels: {channels}")
    
            cv2.imshow(image_filename, image)


    def gen_output():
        pass