# main.py

# +-----------------------------------+
# | HACKERRANK ORCHESTRATE CHALLENGE  |
# +-----------------------------------+
# @link https://www.hackerrank.com/contests/hackerrank-orchestrate-august26/challenges/message-notification-router
# Clone Repo - @link https://github.com/interviewstreet/hackerrank-orchestrate-august26

__author__ = "Jason Monroe (jason@jasonmonroe.com)"
__copyright__ = "Copyright © 2011-2026 Monroe Labs"
__date__ = "2026-08-01"
__version__ = "1.0.0"


# Python Libraries
from dotenv import load_dotenv

load_dotenv("../.env") #../.env"

import sys
import warnings

# Local Libraries
from evaluation.main import run_data_pipeline, run_message_reviewer_pipeline
from src.constants import ARGS_LIST
from src.utils import gen_run_id, show_banner, show_timer, start_timer

 
def run_main_pipeline(args: list):
    print(f"args={args}")

    show_banner("Message Notification Router")

    messages = run_data_pipeline(args)

    # Message Reviewer
    output = run_message_reviewer_pipeline(messages)


    # should interrupt now?

    # whether the message can be batched into a digest,

    # whether it should be muted.
    pass


def parse_args(command_line_args: list[str]) -> dict:
    return {arg.strip("--"): (arg in command_line_args) for arg in ARGS_LIST}


if __name__ == "__main___":
    warnings.filterwarnings("ignore")
    prog_start_time = start_timer()
    
    run_id = gen_run_id()
    print(f"\n----- 🖨️️ START RUN ID: {run_id} 🖨️️ -----")
  
    args = parse_args(sys.argv[1:])
    run_main_pipeline(args)

    show_timer(prog_start_time)
    print(f"\n----- 🖨️️ END RUN ID: {run_id} 🖨️️ -----\n")
