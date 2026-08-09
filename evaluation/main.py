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
from src.utils import get_progress_bar, get_time, log_chat_transcript, show_timer, start_timer


def run_data_pipeline(args: list):
    data_handler = DataHandler(args)

    if args.get("eda"):
        data_handler.describe()

    return data_handler

def run_message_reviewer_pipeline(data_handler) -> list | None:
    messages_df = data_handler.messages
    row_cnt = messages_df.shape[0]

    chat_model = ChatProcessorModel(row_cnt)
    assembler = ContextAssembler(data_handler)

    output_rows = []
    for row in messages_df.itertuples():
        if row.Index >= 0:
            log_chat_transcript("PROMPT_ASSEMBLY", f"Assembling prompt for index: {row.Index}, message ID: {row.message_id}")

            start_time = start_timer()
            prompt = assembler.build_prompt_by_user(row)
            log_chat_transcript("PROMPT_BUILT", prompt) # Logs the exact XML/Text sent to the LLM
      
            response = chat_model.get_response(prompt, row.Index) # response is a list
            log_chat_transcript("LLM RESPONSE", response)
    
            if isinstance(response, dict) and "error" in response or not response:
                print(f"\nNo response was given due to an error.  Breaking loop at index {row.Index}.")
                break

            log_chat_transcript("LLM Response Time", get_time(start_time))
            show_timer(start_time)
           
            time.sleep(PAUSE_TIMER)
            output_rows.append(response)

            log_chat_transcript("Progress Bar", get_progress_bar(row.Index, row_cnt))
            print(get_progress_bar(row.Index, row_cnt))

    # List of output rows that need to be formatted
    return output_rows
    