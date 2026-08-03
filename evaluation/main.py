# code/evaluation/main.py

# Vendor Libraries
import pandas as pd
from scipy.sparse import data

# Local Libraries
from models.chat_processor_model import ChatProcessorModel
from src.context_assembler import ContextAssembler
from src.data_handler import DataHandler
from src.user_analyer import UserAnalyzer


def run_data_pipeline(args: list):
    
    data_handler = DataHandler(args)
    #data_handler.describe()

    return data_handler
    #return data_handler.messages.copy()






def run_message_reviewer_pipeline(data_handler):
    print("\nRunning Message Reviewer Pipeline")

    

    output = ""

    messages_df = data_handler.messages

    dataset = {
        "row_total": messages_df.shape[1]
    }
    
    chat_model = ChatProcessorModel(dataset)

    """

    user = UserAnalyzer(
        data_handler.users, 
        data_handler.group_members, 
        data_handler.business_acounts,
        data_handler.message_history
        )    

    assembler = ContextAssembler({
        "users": data_handler.users,
        "group_members": data_handler.group_members,
        "business_accounts": data_handler.business_accounts,
        "message_history": data_handler.message_history,
        "user_business_history": data_handler.user_business_history,
    })
    """

    import sys

    assembler = ContextAssembler(data_handler)

    for row in messages_df.itertuples():

        print(f"Index: {row.Index} | User ID:  {row.user_id} | MSG ID: {row.message_id} | Media Type: {row.media_type} | Msg TXT: {row.message_text}")
        
        print("\n")
        # Ingestion and Context
        # The Ingestion & Context Pipeline: For each row in messages.csv, aggregate the corresponding user 
        # preferences (users.csv), group role (group_members.csv), business status (business_accounts.csv), 
        # and past chat history (message_history.csv) into a single structured context payload.
        dataset = assembler.build_prompt_by_user(row)

        print(f"message_prompt={dataset["prompt"]}")
        #response = chat_model.get_response(dataset["prompt"])
        #print(f"response = {response}")


        # Formulate output
        output = ""

        #prompt = assembler.create_content(user_dataset)

        sys.exit(0)
        
        if row.Index > 0:
            print(f"row.index = {row.Index}")
            print("break")
            break
        


    # Output CSV

def run_ingestion_contextual_pipeline(users, group_members, business_accounts):
    pass


def run_output_pipeline():
    csv_headers = "message_id,action,message_type,reason,confidence,evidence_message_ids"
