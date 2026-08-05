# code/evaluation/main.py

# Python Libraries
import time
import sys

# Vendor Libraries
import pandas as pd
 
# Local Libraries
from models.chat_processor_model import ChatProcessorModel
from src.constants import PAUSE_TIMER
from src.context_assembler import ContextAssembler
from src.data_handler import DataHandler
from src.utils import get_progress_bar, show_timer, start_timer


def run_data_pipeline(args: list):
    data_handler = DataHandler(args)

    if args.get("eda"):
        data_handler.describe()

    return data_handler


def run_message_reviewer_pipeline(data_handler) -> list | None:
    print("\nRunning Message Reviewer Pipeline")

    messages_df = data_handler.messages
    messages_cnt = messages_df.shape[0]

    chat_model = ChatProcessorModel(messages_cnt)
    assembler = ContextAssembler(data_handler)
    import sys

    output_rows = []
    for row in messages_df.itertuples():
        # @TODO - only test one particular row
        print(f"\nrow.Index={row.Index}")
        if row.Index == 21:
            
            start_time = start_timer()
            prompt = assembler.build_prompt_by_user(row)
            print(f"\nDBG:prompt = {prompt}")
            response = chat_model.get_response(prompt) # response is a list
            print(f"\nDBG: response = {response}")
            show_timer(start_time)
        
            time.sleep(PAUSE_TIMER)
            output_rows.append(response)

            print(get_progress_bar(row.Index, messages_cnt))

    # List of output rows that need to be formatted
    print(f"\nDBG: output_rows={output_rows}")
    #sys.exit(0)
    return output_rows
    