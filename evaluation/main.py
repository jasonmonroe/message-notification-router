# code/evaluation/main.py

# Vendor Libraries
import pandas as pd

# Local Libraries
from models.chat_processor_model import ChatProcessorModel
from src.data_handler import DataHandler


def run_data_pipeline(args: list) -> pd.DataFrame:
    
    data_handler = DataHandler(args)
    #data_handler.describe()

    return data_handler.messages.copy()


def run_message_reviewer_pipeline(messages_df: pd.DataFrame):
    print("\nRunning Message Reviewer Pipeline")

    

    output = ""

    dataset = {
        "row_total": messages_df.shape[1]
    }
    
    #chat_model = ChatProcessorModel(dataset)
    for row in messages_df.itertuples():
        print(f"Index: {row.Index} | User ID:  {row.user_id} | MSG ID: {row.message_id} | Msg TXT: {row.message_text}")
        


    # Output CSV


def run_output_pipeline():
    csv_headers = "message_id,action,message_type,reason,confidence,evidence_message_ids"
