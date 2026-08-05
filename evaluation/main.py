# code/evaluation/main.py

# Python Libraries
import time

# Vendor Libraries
import pandas as pd
 
# Local Libraries
from models.chat_processor_model import ChatProcessorModel
from src.constants import SLEEP_TIMER
from src.context_assembler import ContextAssembler
from src.data_handler import DataHandler
from src.utils import get_progress_bar


def run_data_pipeline(args: list):
    data_handler = DataHandler(args)
    #data_handler.describe()

    return data_handler


def run_message_reviewer_pipeline(data_handler) -> list | None:
    print("\nRunning Message Reviewer Pipeline")

    messages_df = data_handler.messages
    messages_cnt = messages_df.shape[0]
    dataset = {"row_cnt": messages_cnt}

    chat_model = ChatProcessorModel(dataset)
    assembler = ContextAssembler(data_handler)
    import sys

    output_rows = []
    for row in messages_df.itertuples():
        # @TODO - only test one particular row
        if row.Index == 21:
            
            prompt = assembler.build_prompt_by_user(row)
            response = chat_model.get_response(prompt) # response is a list
        
            time.sleep(SLEEP_TIMER)
            output_rows.append(response)

            # Convert LLM response into the proper output to save to CSV file
            #data_handler.save_output(response)

            print(get_progress_bar(row.Index, messages_cnt))

    # List of output rows that need to be formatted
    return output_rows
    