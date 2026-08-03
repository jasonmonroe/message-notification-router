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


def run_message_reviewer_pipeline(data_handler) -> list | None:
    print("\nRunning Message Reviewer Pipeline")

    messages_df = data_handler.messages
    row_total = messages_df.shape[1]
    dataset = {"row_total": row_total}

    chat_model = ChatProcessorModel(dataset)
    assembler = ContextAssembler(data_handler)
    import sys

    output = []
    for row in messages_df.itertuples():
        # @TODO - only test one particular row
        if row.Index == 21:
            
            dataset = assembler.build_prompt_by_user(row)
            #print(f"message_prompt={dataset["prompt"]}")
            response = chat_model.get_response(dataset["prompt"])
            output.append(response)
            #print(f"response = {response}")

            # Convert LLM response into the proper output to save to CSV file
            data_handler.save_output(response)

            print(get_progress_bar(row.Index, row_total))

    # List of output rows that need to be formatted
    return output

    

        