import os, json
import re
import string, random
from openai import OpenAI
from anthropic import Anthropic
from retry import retry
from prompts import *


num_pattern = re.compile(r'\-?\d+\.\d+|\-?\d+')

def write_json(data, filepath, mode = "w"):
    with open(filepath, mode) as f:
        json.dump(data, f)


def read_json(path):
    with open(path, 'r') as f:
      data = json.load(f)
    return data


def generate_random_string(max_length):
    characters = string.ascii_letters + string.digits + string.punctuation
    random_string = ''.join(random.choice(characters) for _ in range(random.randint(1, max_length)))
    return random_string




def get_context(substitute_args, mode = "refine"):

    if mode == "eval":
        context = [{"role": "user", "content": critique_prompt.safe_substitute(**substitute_args)}]
    
    elif mode == "refine":
        context = [{"role": "user", "content": refinement_prompt.safe_substitute(**substitute_args)}]

    elif mode == "ideate":
        context = [{"role": "user", "content": ideate_prompt.safe_substitute(**substitute_args)}]
    
    elif mode == "check_answer":
        context = [{"role": "user", "content": check_answer_prompt.safe_substitute(**substitute_args)}]

    else:
        print(f"{mode} not defined ! stick or either 'refine', 'eval' or 'ideate'")

    return context



class Model():

    def __init__(self, client, model_name, base_url = "", api_key = "") -> None:
        self.client = client
        self.llm = self.init_client(client, base_url = base_url, api_key = api_key)
        self.model_name = model_name
        self.base_url = base_url


    def init_client(self, client, base_url = "", api_key = ""):
        
        if client == "openai":
            return OpenAI(base_url = base_url, api_key = api_key) if base_url and api_key else OpenAI()

        elif client == "anthropic":
            return Anthropic()

    @retry()
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


    def eval_call(self, context, temperature = 0.75):
        response = self.call(context = context, temperature = temperature)
        print(f"doing eval...")
        match = re.search(r'<score>(.*?)<\/score>', response)
        try:
            score = match.group(1)
            print(score)
            
            # remaining_text = re.sub(r'<score>.*?<\/score>', '', response).strip()
            # return remaining_text, int(score)
            return response, int(score)
        
        except Exception as e:
            print(f"could not extract score ! because {e}") 
            print("Trying again ----------------")
            
            # some models cant follow the instruction of putting it within <score> tags.
            score = response.split("Score")
            if len(score) > 1:
                score = score[-1]
                if "/" in score:
                    score = score.split("/")[0]
                
                score = num_pattern.findall(score)
                if score:
                    score = int(score[0])

            else:
                score = 1
            print(f"score: {score}")
            return response, score


    # def 

