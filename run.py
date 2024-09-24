from modules import *
from time import time
from datasets import load_dataset
from config import *
import hashlib
from utils import *


split, level = "test", "2"
dataset = load_dataset("lighteval/MATH", 'all',split = split, trust_remote_code = True)
dataset = dataset.filter(lambda example: example["level"].endswith(level))
dataset.shuffle()


model_name = model_name.split("/")[-1]
save_dir = f"./beam_search/math/{model_name}/{level}_{split}/"
os.makedirs(save_dir, exist_ok = True)
print(model_name, save_dir)


def run(sample, model, if_exists = "skip"):
    query, ground_truth, problem_type = sample["problem"], sample["solution"], sample["type"] # specific to this dataset; 
    print(f"Solving: {query}")
    _file_name = f"{hashlib.md5(str(sample).encode()).hexdigest()}.json"
    save_path = os.path.join(save_dir, _file_name)

    if os.path.exists(save_path):
        if if_exists == "skip":
            print(f"{_file_name} exisits; continuing")
            return 
        
        elif if_exists == "eval_0shot":
            # in case you want to compare with a 0 shot answer of the same question

            save_path = os.path.join(f"./beam_search/math/{model_name}/zeroshot/{level}_{split}/", _file_name)
            responses = {"responses": [model.call(context = [{"role": "user", "content": query}]) for _ in range(3)]}
            responses["query"] = query
            responses["ground_truth"] = ground_truth
            responses["model_name"] = model_name
            responses["problem_type"] = problem_type
            write_json(data = responses, filepath = save_path)
            return 
        
    print(f"doing {_file_name}")
    
    serialized_results = {}
    results = beam_search(query, max_depth = 2, max_children = 2, beam_width = 1, model = model)
    
    responses = [node_result.serialize() for node_result in results]
    # feedback[i] is obtained from the state[i]'s parent node. ie feedback[i] is obtained from critiquing state[i-1] and therefore state[i] is produced from feedback[i].
    serialized_results["responses"] = responses
    serialized_results["query"] = query
    serialized_results["ground_truth"] = ground_truth
    serialized_results["model_name"] = model_name
    serialized_results["problem_type"] = problem_type
    
    
    write_json(serialized_results, save_path)
    print(f"done with {_file_name}")

if __name__ == "__main__":

    # evaluating only on Level 5 problems of lighteval/MATH

    # split = "test"
    # model = Model(client = 'openai', model_name = "gpt-4o-mini")
    model = Model(client = 'openai', model_name = "gemma-math", base_url=f"http://localhost:{port}/v1/", api_key= "token-abc123")
    # model = Model(client = 'openai', model_name = "/mnt/d/work/models/gemma-1.1-2b-it", base_url=f"http://localhost:{port}/v1/", api_key= "token-abc123")
    for i, x in enumerate(dataset):
        print(f"doing {i}")
        run(x, model = model)
        # break

