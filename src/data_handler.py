# src/data_handler.py

# Python Libraries
import cv2
import librosa as lr
import os

# Vendor Libraries
import pandas as pd
import numpy as np
from tinytag import TinyTag

# Local Libraries
from src.constants import AUDIO_DIR, AUDIO_SAMPLE_RATE, DATASET_DIR, IMAGES_DIR


class DataHandler:
    def __init__(self, args: dict):

        # filepaths
        
        self.data = None
        self.df = None

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

        self._use_sample = args.get("sample")

        self._load_text()


    def _load_text(self):
         
        filename = "sample" if self._use_sample else ""
        filename = f"{filename}_messages.csv"

        csv_path = f"{DATASET_DIR}"

        # Text
        self.messages = pd.read_csv(f"{csv_path}{filename}").copy()
         
        self.business_accounts = pd.read_csv(f"{csv_path}business_accounts").copy()
        self.daily_notification_summary = pd.read_csv(f"{csv_path}daily_notification_summary").copy()
        self.group_members = pd.read_csv(f"{csv_path}group_members").copy()
        self.groups = pd.read_csv(f"{csv_path}groups").copy()
        self.images = pd.read_csv(f"{csv_path}images").copy()
        self.message_events = pd.read_csv(f"{csv_path}message_events").copy()
        self.message_history = pd.read_csv(f"{csv_path}message_history").copy()
        self.user_business_history = pd.read_csv(f"{csv_path}user_business_history").copy()
        self.users = pd.read_csv(f"{csv_path}users").copy()
        self.voice_notes = pd.read_csv(f"{csv_path}voice_notes").copy()

        # Media
        self.audios = []
        self.images = []


    def get_csv_file(self, filename: str):
        if filename:
            filepath = f"{DATASET_DIR}{filename}.csv"
            return pd.read_csv(filepath)
        

    def _load_audio(self):
        # @link https://lr.org/doc/latest/index.html
        # @link https://www.comet.com/site/blog/working-with-audio-data-for-machine-learning-in-python/
        
        audio_files = AUDIO_DIR
        for f in audio_files:
            audio_file = lr.load(f, sr=AUDIO_SAMPLE_RATE) #sr= sample rate
            self.audios.append(audio_file)

    def _load_images(self):
        for image_path in os.path(IMAGES_DIR):
            img_array = cv2.imread(image_path) 
            self.images.append(img_array)

    def describe(self) -> None:

        if not self.messages:
            ValueError("🚨 No messages loaded...")
        
        print("# --- 🗒️ Describe Text Data 🗒️ --- #")

        sample_text = "Sample" if self._use_sample else ""

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

        # Media Content
        if self.audios:
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
        else:
            print("⚠️ Warning: No audio files loaded...")


        if self.images:
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
        else:
            print("⚠️ Warning: No images loaded.")

             
            

    def gen_output():
        pass