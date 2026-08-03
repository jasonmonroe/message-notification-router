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

        self._load()

        # Media
        #self.audios = self._load_audio()
        #self.images = self._load_images()

    def _load(self) -> None:
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

            #print(f"ℹ️ Filename: {audio_filename}")
            
            # Load the raw waveform array and sampling rate
            audio_data, sr = lr.load(audio_filepath, sr=AUDIO_SAMPLE_RATE)
            #print(f"ℹ️ Shape: {audio_data.shape}")
            
            # Calculate duration using the loaded array data
            duration = lr.get_duration(y=audio_data, sr=sr)
            #print(f"ℹ️ Duration: {duration:.2f} seconds")

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
                audio_dataset["title"] = audio_tag.title or "Unknown"
                #for key, value in audio_dataset:
                #    print(f"ℹ️ {key.title()}: {value}")

            except Exception:
                print("ℹ️ Title: Could not read metadata")

            audio_files.append(audio_dataset)
            
        return audio_files



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


    def describe_audio(self) -> None:
        if not self.audios:
            print("\n⚠️ Warning: No audio files loaded...")
            return
            
        print("\n# --- 🔈 Describe Audio Data 🔈 --- #")
        print(f"ℹ️  Audio File Count: {len(self.audios)}")
        
        audio_filenames = os.listdir(AUDIO_DIR)
        print(f"audio_files={audio_filenames}")
        
        for idx, audio in enumerate(self.audios):
            #audio_filename = audio_filenames[idx]
            #audio_filepath = os.path.join(AUDIO_DIR, audio_filename) # Get full path for TinyTag
            #audio_name, _ = os.path.splitext(audio_filename)
            #audio_tag = TinyTag.get(audio_filepath)
            
            print(f"\n🔈 Audio File {idx}")

            for key, value in audio:
                print(f"ℹ️ {key.title()}: {value}")

            #print(f"\tℹ️ Filename: {audio_name}")
            #print(f"\tℹ️ Shape: {audio.shape}") # Works now because audio is an array
            #print(f"\tℹ️ Duration: {lr.get_duration(y=audio, sr=AUDIO_SAMPLE_RATE):.2f} seconds")
            #print(f"\tℹ️ Title: {audio_tag.title}")
               
    def describe_images(self) -> None:

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
    
            cv2.imshow(image_filename, image)

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
        