from modules import *
from time import time
from datasets import load_dataset
from config import *
import hashlib

split = "train"
dataset = load_dataset("lighteval/MATH", 'all',split = split, trust_remote_code = True)
dataset = dataset.filter(lambda example: example["level"].endswith("5"))
dataset.shuffle()

save_dir = "./beam_search/math/"
os.makedirs(save_dir, exist_ok = True)


def write_json(data, filepath, mode = "w"):
    with open(filepath, mode) as f:
        json.dump(data, f)


def read_json(path):
    with open(path, 'r') as f:
      data = json.load(f)
    return data


def run(sample):
    query, ground_truth, problem_type = sample["problem"], sample["solution"], sample["type"] # specific to this dataset; 

    _file_name = f"{hashlib.md5(str(sample).encode()).hexdigest()}.json"
    save_path = os.path.join(save_dir, _file_name)

    if os.path.exists(save_path):
        print(f"{_file_name} exisits; continuing")
        return 

    print(f"doing {_file_name}")
    # print(f"query: {query}")
    # print()
    # print(f"answer: {ground_truth}")

    serialized_results = []
    results = beam_search(query, max_depth = 3, max_children = 2, beam_width = 1)
    for node_result in results:
        result = node_result.serialize()
        
        result["query"] = query
        result["ground_truth"] = ground_truth
        result["model_name"] = model_name
        result["problem_type"] = problem_type

        serialized_results.append(result)
    
    write_json(serialized_results, save_path)
    print(f"done with {_file_name}")

if __name__ == "__main__":

    # evaluating only on Level 5 problems of lighteval/MATH

    # split = "test"
    dataset.map(run, num_proc = 16)


