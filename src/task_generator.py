from openai import OpenAI
import os
from scripts.gpt_caller import *
caller=gpt_caller(api_key=os.environ.get("OPENAI_API_KEY"))
caller.create_prompt(["What is the capital of France?"])
caller.call()