from openai import OpenAI
from prompts import *
from retry import retry
from config import *
from utils import *
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
load_dotenv()


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

    # query = "Solve the following math equation. If x = 3, y = 11, z = 7; compute the value of (x+y-z)^2"
    # query = "Let $a,$ $b,$ $c,$ $d$ be positive real numbers such that\n\\begin{align*}\n(a + b)(c + d) &= 143, \\\\\n(a + c)(b + d) &= 150, \\\\\n(a + d)(b + c) &= 169.\n\\end{align*}Find the smallest possible value of $a^2 + b^2 + c^2 + d^2.$"
    # query = "The expression $\\sqrt{(\\sqrt{56})(\\sqrt{126})}$ can be simplified to $a\\sqrt b$, where $a$ and $b$ are integers and $b$ is not divisible by any perfect square greater than 1. What is $a+b$?"
    # query = "The lengths, in order, of four consecutive sides of an equiangular hexagon are 1, 7, 2 and 4 units, respectively. What is the sum of the lengths of the two remaining sides?"
    # query = "Let \\[f(x) = \\left\\{\n\\begin{array}{cl}\n\\frac{x}{21} & \\text{ if }x\\text{ is a multiple of 3 and 7}, \\\\\n3x & \\text{ if }x\\text{ is only a multiple of 7}, \\\\\n7x & \\text{ if }x\\text{ is only a multiple of 3}, \\\\\nx+3 & \\text{ if }x\\text{ is not a multiple of 3 or 7}.\n\\end{array}\n\\right.\\]If $f^a(x)$ means the function is nested $a$ times (for example, $f^2(x)=f(f(x))$), what is the smallest value of $a$ greater than 1 that satisfies $f(2)=f^a(2)$?"
    query = "Factor the following expression: $145b^2 +29b$."
    # make sure to have your keys in .env 
    # model = Model(client = "openai", model_name = "gpt-4o-mini")
    # model = Model(client = "anthropic", model_name = "claude-3-5-sonnet-20240620")

    # ensure that vLLM is running in the background for this to work. Change port and api_key as needed
    model = Model(client = "openai", base_url = base_url, api_key = api_key, model_name = model_name)
    # model = Model(client = "openai", model_name = "gpt-4o-mini")
    # model = Model(client = "anthropic", model_name = "claude-3-haiku-20240307")
    

    # model = Model(client = "openai", base_url = base_url, api_key = api_key, model_name = "/mnt/d/work/models/gemma-1.1-2b-it")

    result = beam_search(query = query, max_depth = 2, max_children = 3, beam_width=1, model = model)
    print(result)
    print()
    print("-----")
    print()