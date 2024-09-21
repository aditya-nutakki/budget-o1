import os, json
from prompts import *
from dotenv import load_dotenv
from utils import *
from config import *
from modules import *
from datasets import load_dataset
import hashlib


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
    # return data


def get_dpo_data(tuner_model, student_model, split, level = "3", if_exists = "skip"):
    dataset = load_my_data(split = split, level = level)
    preference_data = {}
    
    save_dir = f"./beam_search/math/{tuner_model.model_name}/{level}_{split}/"
    os.makedirs(save_dir, exist_ok = True)
    
    for i, sample in enumerate(dataset):
        query, ground_truth, problem_type = sample["problem"], sample["solution"], sample["type"] # specific to this dataset; 
        _file_name = f"{hashlib.md5(str(sample).encode()).hexdigest()}.json"
        save_path = os.path.join(save_dir, _file_name)
        print(f"Going to save it to {save_path}")
        if os.path.exists(save_path):
            if if_exists == "skip":
                continue

        preference_data["prompt"] = []
        preference_data["chosen"] = []
        preference_data["rejected"] = []
        preference_data["level"] = []
        
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

                
        
        
        """
            eval:
                1. 
        """

        write_json(preference_data, filepath = save_path)
        print(f"done")

        break






if __name__ == "__main__":
    # tuner = Tuner(client = "anthropic", model_name = "claude-3-haiku-20240307")
    save_dir = "./beam_search/"

    tuner_model = Model(client = "openai", model_name = "gpt-4o-mini")

    # make sure your server on vLLM is running
    port = 8000
    base_url = f"http://localhost:{port}/v1"
    api_key = "token-abc123"
    student_model = Model(client = "openai", base_url = base_url, api_key = api_key, model_name = model_name)
    # print(tuner.call(context = [{"role":"user", "content": "hi how are you "}]))
    
    get_dpo_data(tuner_model, student_model, split = "train")





