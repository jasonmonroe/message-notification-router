# code/evaluation/main.py


# Local Libraries
from models.chat_processor_model import ChatProcessorModel
from src.data_handler import DataHandler


def run_data_pipeline(args: list):
    
    data_handler = DataHandler(args)

    return data_handler.messages


def run_message_reviewer_pipeline(messages):
    print("Running Message Reviewer Pipeline")

    output = ""

    dataset = {}
    chat_model = ChatProcessorModel(dataset)
    for message in messages:
        pass


    # Output CSV


def run_output_pipeline():
    csv_headers = "message_id,action,message_type,reason,confidence,evidence_message_ids"
