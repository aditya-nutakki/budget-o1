import os, json
from prompts import *
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
from utils import *


load_dotenv()

"""
The whole premise of this is to get winning and losing responses for generating chain of thoughts

Winning response would be computed by the bigger model ex: gpt-4o, sonnet 3.5 etc
losing response would be whatever computed by your model

These (prompt, winning_response, losing_response) should be done for both generating cot as well as the feedback


1. Pass the question, ground truth and the cot generated from your model and ask the bigger model to generate the chain of thought which is along the lines of whatever was generated
2. To generate the winning and losing responses for the feedback. You must pass the question, ground truth, newly generated cot and the wrong feedack to generate the correct feedback
3. what about the part which incorporates feedback to  generated something (?)
3. Similar to point2, take the generated cot/plan, give the newly generated feedback, the wrong cot from there to generate the correct COT

4. The ultimate goal of this is to only refine the feedback, new idea, incorporation of new idea into the pipeline - figure it out !
"""

class Tuner():

    def __init__(self, client, model_name, base_url = "", api_key = "") -> None:
        self.client = client
        self.llm = self.init_client(client, base_url = base_url, api_key = api_key)
        self.model_name = model_name
        self.base_url = base_url
        # if not api_key:
        #     self.api_key = os.getenv()

    def call(self, context, temperature = 0.75):
        if self.client == "openai":
            return self.llm.chat.completions.create(
                # model = "gpt-4o-mini",
                model = self.model_name,
                messages = context,
                temperature = temperature
            ).choices[0].message.content

        elif self.client == "anthropic":
            return self.llm.messages.create(
                model = self.model_name,
                max_tokens = 4096 - 1,
                # system = system,
                # tools = tools,
                messages = context,
                temperature = temperature,
                stream = False
                ).content[0].text


    def init_client(self, client, base_url = "", api_key = ""):
        
        if client == "openai":
            return OpenAI(base_url = base_url, api_key = api_key) if base_url and api_key else OpenAI()

        elif client == "anthropic":
            return Anthropic()
        

def process_data(path):
    data = read_json(path)
    pairs = {}

    





if __name__ == "__main__":
    # tuner = Tuner(client = "anthropic", model_name = "claude-3-haiku-20240307")
    save_dir = "./beam_search/"
    tuner = Tuner(client = "openai", model_name = "gpt-4o-mini")
    # print(tuner.call(context = [{"role":"user", "content": "hi how are you "}]))






