import os, json, re
from openai import OpenAI
from prompts import *
from retry import retry
from config import *
from utils import *
from functools import partial
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()


if model_name not in ["gpt-4o", "gpt-4o-mini"]:
    port = 8000
    base_url = f"http://localhost:{port}/v1"
    api_key = "token-abc123"

    client = OpenAI(base_url = base_url, api_key = api_key)

else:
    client = OpenAI()

# num_pattern = re.compile(r'\-?\d+\.\d+|\-?\d+')


# def raw_call(context, temperature = 0.5):
#     response = client.chat.completions.create(
#         # model = "gpt-4o-mini",
#         model = model_name,
#         messages = context,
#         temperature = temperature,
#     )
#     return response.choices[0].message.content


# @retry()
# def call_llm(context, temperature = 0.9, mode = "refine"):

#     response = raw_call(context = context, temperature = temperature)
#     print(response)

#     # extract score through regex and return the number as an int
#     if mode == "eval":
#         print(f"doing eval...")
#         match = re.search(r'<score>(.*?)<\/score>', response)
#         try:
#             score = match.group(1)
#             print(score)
            
#             remaining_text = re.sub(r'<score>.*?<\/score>', '', response).strip()
#             return remaining_text, int(score)
        
#         except Exception as e:
#             print(f"could not extract score ! because {e}") 
#             print("Trying again ----------------")
            
#             # some models cant follow the instruction of putting it within <score> tags.
#             score = response.split("Score")
#             if len(score) > 1:
#                 score = score[-1]
#                 if "/" in score:
#                     score = score.split("/")[0]
                
#                 score = num_pattern.findall(score)
#                 if score:
#                     score = int(score[0])

#             else:
#                 score = 1
#             print(f"score: {score}")
#             return response, score

#     return response



# def get_context(substitute_args, mode = "refine"):

#     if mode == "eval":
#         context = [{"role": "user", "content": critique_prompt.safe_substitute(**substitute_args)}]
    
#     elif mode == "refine":
#         context = [{"role": "user", "content": refinement_prompt.safe_substitute(**substitute_args)}]

#     elif mode == "ideate":
#         context = [{"role": "user", "content": ideate_prompt.safe_substitute(**substitute_args)}]
#     else:
#         print(f"{mode} not defined ! stick or either 'refine', 'eval' or 'ideate'")

#     return context


class Node:
    def __init__(self, state, parent = [], score = 1, feedback = ""):
        self.state = state
        self.parent = parent
        self.score = score
        self.feedback = feedback
        

    def __repr__(self):
        return f"Node(state={self.state}, score={self.score})"
    

    def expand(self, query, max_children, model):
        
        def expand_once(x):
            substitute_args = {"query": query, "thoughts": self.state, "ans_format": "score"}
            # feedback, score = call_llm(context = get_context(substitute_args = substitute_args, mode = "eval"), mode = "eval")
            feedback, score = model.eval_call(context = get_context(substitute_args = substitute_args, mode = "eval"))

            # print(self.parent)
            if self.parent:
                parent_score = self.parent.score
            else:
                parent_score = 1 # only applicable for the root node


            score *= parent_score
            substitute_args = {"query": query, "feedback": feedback, "solution": self.state, "ans_format": "solution"}
            # refinement = call_llm(context = self.get_context(query = query, mode = "refine"), mode = "refine")
            # refinement = call_llm(context = get_context(substitute_args = substitute_args, mode = "refine"), mode = "refine")
            refinement = model.call(context = get_context(substitute_args = substitute_args, mode = "refine"))
            return Node(state = refinement, parent = self, score = score, feedback = feedback)


        with ThreadPoolExecutor(max_workers = 5) as executor:
            self.children = list(executor.map(expand_once, list(range(max_children))))
        
        # self.children = []
        # for _ in range(max_children):
        #     self.children.append(expand_once(_))


    def path(self):
        
        node, path_back = self, []
        while node:
            # path_back.append(node.state)
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))[1:] # literal top to bottom path ie [root, layer1_node, layer2_node, final_node]; skipping the 0th node as it is the root node which contains the query


    def serialize(self):
        # should serialize the object so it can be saved as a JSON
        data = {}
        path = self.path()

        states, scores, feedback = [], [], []
        for i, node in enumerate(path):
            states.append(node.state)
            scores.append(node.score)
            feedback.append(node.feedback)
        
        data["states"] = states
        data["scores"] = scores
        data["feedback"] = feedback

        return data


# def get_init_ideas(query):
#     return raw_call([{"role": "user", "content": ideate_prompt.safe_substitute({'query': query, 'ans_format': 'steps'})}])

# def beam_search(root_node, max_depth, max_children, beam_width):
def beam_search(query, max_depth, max_children, beam_width, model):
    
    def get_init_ideas(query):
        return model.call([{"role": "user", "content": ideate_prompt.safe_substitute({'query': query, 'ans_format': 'steps'})}])

    root_node = Node(state = query, parent = [])

    with ThreadPoolExecutor(max_workers = 10) as executor:
        _nodes = list(executor.map(get_init_ideas, [query]*max_children))

    _nodes = [Node(state = node, parent = root_node) for node in _nodes]

    print(_nodes)
    print("-----")
    
    visited = _nodes

    curr_depth = 0
    while visited and curr_depth < max_depth:

        current_nodes = visited
        visited = []

        while current_nodes:
            
            node = current_nodes.pop(0)
            node.expand(query = query, max_children = max_children, model = model)
            print(f"Node's children are: {node.children}")
            print()
            for child_node in node.children:
                visited.append(child_node)
            
        # visited.sort(key = lambda n: n.val, reverse = True)[:beam_width]
        visited = sorted(visited, key = lambda n: n.score, reverse = True)[:beam_width]
        curr_depth += 1

    print("----")
    return visited


if __name__ == "__main__":

    """
    Things to do:
        1. Quick break if answer is already achieved.
        2. need to rewrite 'critique' prompt # done; improved performance
        3. There _might_ be a bug wrt scoring; take a look 
    """


    # query = "Solve the following math equation. If x = 3, y = 11, z = 7; compute the value of (x+y-z)^2"
    query = "Let $a,$ $b,$ $c,$ $d$ be positive real numbers such that\n\\begin{align*}\n(a + b)(c + d) &= 143, \\\\\n(a + c)(b + d) &= 150, \\\\\n(a + d)(b + c) &= 169.\n\\end{align*}Find the smallest possible value of $a^2 + b^2 + c^2 + d^2.$"

    # best_path = beam_search(root_node = Node(state = query, parent = []), max_depth = 3, max_children = 2, beam_width=1)
    best_path = beam_search(query = query, max_depth = 3, max_children = 2, beam_width=1)
    print("Best path found:", best_path)
    print()
    print("-----")
    print()
    for v in best_path:
        # x = v.path()
        # for _node in x:
        #     print(_node.state, _node.score)
        # print()
        data = v.serialize()
        with open("./dummy.json", "w") as f:
            json.dump(data, f)


