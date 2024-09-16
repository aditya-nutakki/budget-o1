import os, json
import re
import random
import string
from openai import OpenAI
from prompts import *
from retry import retry

from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv

load_dotenv()

port = 8000
base_url = f"http://localhost:{port}/v1"
api_key = "token-abc123"

# client = OpenAI(base_url = base_url, api_key = api_key)
client = OpenAI()


def generate_random_string(max_length):
    characters = string.ascii_letters + string.digits + string.punctuation
    random_string = ''.join(random.choice(characters) for _ in range(random.randint(1, max_length)))
    return random_string


def raw_call(context, temperature = 0.5):
    response = client.chat.completions.create(
        # model = "my-model",
        model = "gpt-4o-mini",
        messages = context,
        temperature = temperature,
    )
    return response.choices[0].message.content


@retry()
def call_llm(context, temperature = 0.9, mode = "refine"):

    response = raw_call(context = context, temperature = temperature)
    print(response)

    # extract score through regex and return the number as an int
    if mode == "eval":
        print(f"doing eval...")
        match = re.search(r'<score>(.*?)<\/score>', response)
        try:
            score = match.group(1)
            print(score)
            
            remaining_text = re.sub(r'<score>.*?<\/score>', '', response).strip()
            return remaining_text, int(score)
        
        except Exception as e:
            print(f"could not extract score ! because {e}") 
            
            print()
            print()
            print("--------------------")

            return response, 1

    elif mode == "refine":
        print(f"doing refinement...")
        # match = re.search(r'<solution>(.*?)<\/solution>', response)
        # try:
        #     solution = match.group(1)
        #     print(solution)  # Output: 8
        #     return solution
        
        # except Exception as e:
        #     print(f"could not extract solution ! because {e}") 
            
        #     print()
        #     print()
        #     print("--------------------")

        #     return response
        return response


    elif mode == "ideate":
        print(f"doing ideation")
        match = re.search(r'<steps>(.*?)<\/steps>', response)
        try:
            solution = match.group(1)
            print(solution)  # Output: 8
            return solution
        
        except Exception as e:
            print(f"could not extract solution ! because {e}") 
            
            print()
            print()
            print("--------------------")

            return response

    else:
        print(f"{mode} not defined; stick to either 'refine' or 'eval'")
        return response
    

class Node:
    def __init__(self, state, parent = []):
        self.state = state
        self.parent = parent
        self.score, self.llm_score = 1, 1
        self.feedback = ""
        self.context = []
        

    def __repr__(self):
        # return f"Node(state={self.state}, score={self.score}, parent={self.parent})"
        return f"Node(state={self.state}, score={self.score})"


    def get_context(self, query, mode = "refine"):

        if mode == "eval":
            context = [{"role": "user", "content": critique_prompt.safe_substitute({
                "query": query,
                # "thoughts": self.parent.state,
                "thoughts": self.state,
                "ans_format": "score"
            })}]
        
        elif mode == "refine":
            context = [{"role": "user", "content": refinement_prompt.safe_substitute({
                "query": query,
                # "hints": self.parent.feedback,
                "hints": self.feedback,
                "solution": self.state,
                "ans_format": "solution"
            })}]

        elif mode == "ideate":
            context = [[{"role": "user", "content": ideate_prompt.safe_substitute({
                "query": query,
                # "hints": self.parent.feedback,
                "hints": self.feedback,
                "solution": self.state,
                "ans_format": "solution"
            })}]]
        else:
            print(f"{mode} not defined ! stick or either 'refine', 'eval' or 'ideate'")

        return context


    def expand(self, query, max_children):
        
        def expand_once(x):
            feedback, score = call_llm(context = self.get_context(query = query, mode = "eval"), mode = "eval")
            
            self.feedback = feedback
            self.llm_score = score
            print(self.parent)
            if self.parent:
                parent_score = self.parent.score
            else:
                parent_score = 1

            self.score = self.llm_score * parent_score
            print(self.score)

            refinement = call_llm(context = self.get_context(query = query, mode = "refine"), mode = "refine")
            return Node(state = refinement, parent = self)

        # wrt max_children; max_depth
        # self.children = [Node(state = generate_random_string(15), parent = self) for _ in range(max_children)] # call this in parallel

        # with ThreadPoolExecutor(max_workers = 5) as executor:
        #     self.children = list(executor.map(expand_once, list(range(max_children))))
        self.children = []
        for _ in range(max_children):
            self.children.append(expand_once(_))


    def eval(self):
        
        # context = self._get_parent_context()
        # feedback, llm_score = call_llm(context = [], mode = "eval") # should return a single value which denotes the q-value of the node
        
        # self.parent.feedback = feedback
        # llm_score will have to be extracted; 
        # self.state = new_response generated;
        # ultimately; context has to be built 

        parent_score = self.parent.score
        # self.score = llm_score * self.parent.score
        self.score = self.llm_score * parent_score
        
        
    def path(self):
        
        node, path_back = self, []
        while node:
            # path_back.append(node.state)
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back)) # literal top to bottom path ie [root, layer1_node, layer2_node, final_node]


    def serialize():
        # should serialize the object so it can be saved as a JSON
        pass




def beam_search(root_node, max_depth, max_children, beam_width):

    query = root_node.state

    # first_refinement = call_llm([{"role": "user", "content": ideate_prompt.safe_substitute({'query': query, 'ans_format': 'solution'})}], mode = "refine")
    first_refinement = raw_call([{"role": "user", "content": ideate_prompt.safe_substitute({'query': query, 'ans_format': 'steps'})}])
    
    print(f"first refinement: {first_refinement}")
    print()
    root_node.state = first_refinement
    # root_node.feedback = first_iteration
    
    visited = [root_node]
    # all_nodes = [root_node]

    curr_depth = 0
    while visited and curr_depth < max_depth:

        current_nodes = visited
        visited = []

        while current_nodes:
            
            node = current_nodes.pop(0)
            node.expand(query = query, max_children = max_children)
            print(f"Node's children are: {node.children}")
            print()
            for child_node in node.children:
                # child_node.eval() # adds a score property to it
                visited.append(child_node)
                # all_nodes.append(child_node)
            
        # visited.sort(key = lambda n: n.val, reverse = True)[:beam_width]
        visited = sorted(visited, key = lambda n: n.score, reverse = True)[:beam_width]

        curr_depth += 1
        print(curr_depth)

    print(visited)
    # print("----")
    # print(all_nodes, len(all_nodes))
    print("----")
    # print(sorted(all_nodes, key = lambda n: n.score, reverse = True)[:beam_width])

    return visited


if __name__ == "__main__":

    """
    Things to do:
        1. Quick break if answer is already achieved.
        2. generate diversity from 'ideate' mode - it is fundamentally wrong to generate the idea only once and ask it to rate it.
        3. refinement_prompt has it such that the solution must be enclosed within tags and we extract those tags, consider to not do this
        4. node.eval() must be done within the expand node itself. There is no point to it doing it outside
    """


    # query = "Solve the following math equation. If x = 3, y = 11, z = 7; compute the value of (x+y-z)^2"
    query = "Let $a,$ $b,$ $c,$ $d$ be positive real numbers such that\n\\begin{align*}\n(a + b)(c + d) &= 143, \\\\\n(a + c)(b + d) &= 150, \\\\\n(a + d)(b + c) &= 169.\n\\end{align*}Find the smallest possible value of $a^2 + b^2 + c^2 + d^2.$"

    best_path = beam_search(root_node = Node(state = query, parent = []), max_depth = 4, max_children = 2, beam_width=1)
    print("Best path found:", best_path)
    print()
    print("-----")
    print()
    for v in best_path:
        x = v.path()
        for _node in x:
            print(_node.state, _node.score)
        print()

