# code/evaluation/main.py

# Vendor Libraries
import pandas as pd
 
# Local Libraries
from models.chat_processor_model import ChatProcessorModel
from src.context_assembler import ContextAssembler
from src.data_handler import DataHandler
from src.utils import get_progress_bar


def run_data_pipeline(args: list):
    
    data_handler = DataHandler(args)
    #data_handler.describe()

    return data_handler


def run_message_reviewer_pipeline(data_handler):
    print("\nRunning Message Reviewer Pipeline")

    messages_df = data_handler.messages
    row_total = messages_df.shape[1]

    dataset = {"row_total": row_total}
    chat_model = ChatProcessorModel(dataset)


    import sys

    assembler = ContextAssembler(data_handler)

    for row in messages_df.itertuples():
        

        if row.Index == 21:
            # Ingestion and Context
            # The Ingestion & Context Pipeline: For each row in messages.csv, aggregate the corresponding user 
            # preferences (users.csv), group role (group_members.csv), business status (business_accounts.csv), 
            # and past chat history (message_history.csv) into a single structured context payload.
            dataset = assembler.build_prompt_by_user(row)

            print(f"message_prompt={dataset["prompt"]}")
        
            response = chat_model.get_response(dataset["prompt"])
            print(f"response = {response}")

            """
            response = {
                'message_id': 'sample_msg_001', 
                'action': 'notify', 
                'message_type': 'utility_urgent', 
                'confidence': 0.98,
                'evidence_message_ids': ['message_0129', 'message_0215'],
                'reason': (
                    'Highly time-sensitive notice regarding water supply with a 20-minute '
                    'window for action, sent to an unmuted society notices group where '
                    'the user is an admin.'
                )
            }
            """

            # Convert LLM response into the proper output to save to CSV file
            data_handler.save_output(response)

            print(get_progress_bar(row.Index, row_total))
        