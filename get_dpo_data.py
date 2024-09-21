import os, json
from prompts import *
from dotenv import load_dotenv
from utils import *
from config import *
from modules import *
from datasets import load_dataset
import hashlib
from concurrent.futures import ThreadPoolExecutor
from time import sleep

load_dotenv()

"""
The whole premise of this is to get winning and losing responses for generating chain of thoughts

Winning response would be computed by the bigger model ex: gpt-4o, sonnet 3.5 etc
losing response would be whatever computed by your model

These (prompt, winning_response, losing_response) should be done for both generating cot as well as the feedback

Then train another model with DPO.
"""


model_name = model_name.split("/")[-1]

def load_my_data(split = "train", level = "5"):
    # split, level = "train", "5"
    dataset = load_dataset("lighteval/MATH", 'all',split = split, trust_remote_code = True)
    dataset = dataset.filter(lambda example: example["level"].endswith(level))
    dataset.shuffle()
    return dataset


def update_data(data, prompt, chosen, rejected):
    data["prompt"].append(prompt)
    data["chosen"].append(chosen)
    data["rejected"].append(rejected)


def get_dpo_data(tuner_model, student_model, split, level = "3", if_exists = "skip"):


    def _do_one_sample(sample):
        preference_data = {}
        if isinstance(sample, str):
            sample = json.loads(sample)
    
        query, ground_truth, problem_type = sample["problem"], sample["ground_truth"], sample["problem_type"] # specific to this dataset; 
        _file_name = f"{hashlib.md5(str(sample).encode()).hexdigest()}.json"
        save_path = os.path.join(save_dir, _file_name)
        print(f"Going to save it to {save_path}")
        sleep(1.5)
        
        if os.path.exists(save_path):
            if if_exists == "skip":
                print(f"Skipping {save_path}")
                return

        preference_data["prompt"] = []
        preference_data["chosen"] = []
        preference_data["rejected"] = []
        preference_data["level"] = level
        preference_data["ground_truth"] = ground_truth
        preference_data["query"] = query
        
        # try to get both the tuner as well as the student model to run together

        # one way is to call student model right after tuner model. 
        # another way is to first finish off the tuner model and then retrace that tree with student model to obtain y_losing and y_winnnig
        tuner_results = beam_search(query = query, max_depth = 2, max_children = 2, beam_width = 1, model = tuner_model)
        tuner_results = [node_result.serialize() for node_result in tuner_results]
        
        print(f"Done with tuner model results. Now doing it for student model ...")
        for tuner_result in tuner_results:
            tuner_idea = tuner_result["states"][0]
            ideate_context = get_context(substitute_args = {"query": query}, mode = "ideate")
            student_model_idea = student_model.call(context = ideate_context)

            # preference_data["chosen"] = tuner_idea
            # preference_data["rejected"] = student_model_idea
            # preference_data["prompt"] = ideate_context[0]["content"]

            update_data(preference_data, prompt = ideate_context[0]["content"], chosen = tuner_idea, rejected = student_model_idea)

            num_states = len(tuner_result['states'])
            for i in range(1, num_states):
                # feedback[i] is obtained from state[i - 1]
                # state[0] is the initial idea generated
                # states[i] = states[i-1] + feedback[i]
                
                # tuner_solution = tuner_result["states"][i]
                critique_context = get_context(substitute_args = {"query": query, "thoughts": tuner_result["states"][i - 1], "ans_format": "score"}, mode = "eval")
                student_model_critique = student_model.call(context = critique_context)
                update_data(data = preference_data, prompt = critique_context[0]["content"], chosen = tuner_result["feedback"][i], rejected = student_model_critique)
                

                refine_context = get_context(substitute_args = {"query": query, "solution": tuner_result["states"][i - 1], "feedback": tuner_result["feedback"][i]}, mode = "refine")
                student_model_refinement = student_model.call(context = refine_context)
                update_data(data = preference_data, prompt = refine_context[0]["content"], chosen = tuner_result["states"][i], rejected = student_model_refinement)


        write_json(preference_data, filepath = save_path)
        print(f"done")


    dataset = load_my_data(split = split, level = level)
    
    # save_dir = f"./beam_search/math/{tuner_model.model_name}/{level}_{split}/"
    save_dir = f"./beam_search/math/dpo/{tuner_model.model_name}/{level}_{split}/"
    os.makedirs(save_dir, exist_ok = True)

    num_concurrent = 8
    for i in range(0, len(dataset), num_concurrent):
        samples = dataset[i : i + num_concurrent]
        _samples = []
        for problem, ground_truth, problem_type in zip(samples["problem"], samples["solution"], samples["type"]):
            _samples.append({"problem": problem, "ground_truth": ground_truth, "problem_type": problem_type})

        with ThreadPoolExecutor(max_workers = num_concurrent) as executor:
            _ = list(executor.map(_do_one_sample, _samples))


if __name__ == "__main__":
    # tuner = Tuner(client = "anthropic", model_name = "claude-3-haiku-20240307")
    save_dir = "./beam_search/"

    tuner_model = Model(client = "openai", model_name = "gpt-4o-mini")
    # tuner_model = Model(client = "openai", model_name = "gpt-4o") # use this for harder problems - level 4 or level 5
    # tuner_model = Model(client = "anthropic", model_name = "claude-3-5-sonnet-20240620") # use this for harder problems - level 4 or level 5
    
    # make sure your server via vLLM is running
    port = 8000
    base_url = f"http://localhost:{port}/v1"
    api_key = "token-abc123"
    student_model = Model(client = "openai", base_url = base_url, api_key = api_key, model_name = model_name)
    # print(tuner.call(context = [{"role":"user", "content": "hi how are you "}]))
    
    get_dpo_data(tuner_model, student_model, split = "train", level = "2")

