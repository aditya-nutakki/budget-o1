from modules import *
from time import time
from datasets import load_dataset
from config import *
import hashlib
from utils import *


split, level = "train", "3"
dataset = load_dataset("lighteval/MATH", 'all',split = split, trust_remote_code = True)
dataset = dataset.filter(lambda example: example["level"].endswith(level))
dataset.shuffle()


model_name = model_name.split("/")[-1]
save_dir = f"./beam_search/math/{model_name}/{level}_{split}/"
# save_dir = f"./beam_search/math/zeroshot/{level}_{split}/"
os.makedirs(save_dir, exist_ok = True)
print(model_name, save_dir)


def run(sample, model, if_exists = ""):
    query, ground_truth, problem_type = sample["problem"], sample["solution"], sample["type"] # specific to this dataset; 

    _file_name = f"{hashlib.md5(str(sample).encode()).hexdigest()}.json"
    save_path = os.path.join(save_dir, _file_name)

    if os.path.exists(save_path):
        # print(f"{_file_name} exisits; continuing")
        # return 
        # save_path = os.path.join(f"./beam_search/math/{model_name}/zeroshot/{level}_{split}/", _file_name)
        # # response = raw_call(context = [{"role": "user", "content": query}])
        # # responses = {"responses": [raw_call(context = [{"role": "user", "content": f"query: {query}\n\nexamples are: {math4shot_prompt}"}]) for _ in range(3)]}
        # responses = {"responses": [raw_call(context = [{"role": "user", "content": query}]) for _ in range(3)]}
        # responses["query"] = query
        # responses["ground_truth"] = ground_truth
        # responses["model_name"] = model_name
        # responses["problem_type"] = problem_type
        # write_json(data = responses, filepath = save_path)
        return 

    # return
    print(f"doing {_file_name}")
    # print(f"query: {query}")
    # print()
    # print(f"answer: {ground_truth}")

    serialized_results = {}
    results = beam_search(query, max_depth = 3, max_children = 3, beam_width = 2, model = model)
    # for node_result in results:
    #     result = node_result.serialize()    
        
    responses = [node_result.serialize() for node_result in results]
    # feedback[i] is obtained from the state[i]'s parent node. ie feedback[i] is obtained from critiquing state[i-1] and therefore state[i] is produced from feedback[i].
    serialized_results["responses"] = responses
    serialized_results["query"] = query
    serialized_results["ground_truth"] = ground_truth
    serialized_results["model_name"] = model_name
    serialized_results["problem_type"] = problem_type
    
    # serialized_results.append(result)
    
    write_json(serialized_results, save_path)
    print(f"done with {_file_name}")

if __name__ == "__main__":

    # evaluating only on Level 5 problems of lighteval/MATH

    # split = "test"
    model = Model(client = 'openai', model_name = "gpt-4o-mini")
    # expanded_run = partial(run, model = model)
    # dataset.map(expanded_run, num_proc = 8)
    for i, x in enumerate(dataset):
        print(f"doing {i}")
        run(x, model = model)
        break

