import os, json
import re
import random
import string

port = 8000
base_url = f"http://localhost:{port}/v1"


def generate_random_string(max_length):
    characters = string.ascii_letters + string.digits + string.punctuation
    random_string = ''.join(random.choice(characters) for _ in range(random.randint(1, max_length)))
    return random_string


class Node:
    def __init__(self, state, parent = []):
        self.state = state
        self.parent = parent
        
        self.context = []


    def __repr__(self):
        # return f"Node(state={self.state}, score={self.score}, parent={self.parent})"
        return f"Node(state={self.state}, score={self.score})"


    def expand(self, max_children):
        # wrt max_children; max_depth
        self.children = [Node(state = generate_random_string(15), parent = self) for _ in range(max_children)]
        

    def eval(self):
        # should return a single value which denotes the q-value of the node
        # self.score = random.randint(1, 10)
        
        llm_score = random.randint(1, 10) # 
        parent_score = self.parent.score if self.parent else 1.0
        self.score = llm_score * parent_score

        
    def path(self):
        
        node, path_back = self, []
        while node:
            # path_back.append(node.state)
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))



def beam_search_v2(root_node, max_depth, max_children, beam_width):

    root_node.score = 1.0

    visited = [root_node]
    all_nodes = [root_node]

    curr_depth = 0
    while visited and curr_depth < max_depth:

        current_nodes = visited
        visited = []

        while current_nodes:
            
            node = current_nodes.pop(0)
            node.expand(max_children = max_children)

            for child_node in node.children:
                child_node.eval() # adds a score property to it
                visited.append(child_node)
                all_nodes.append(child_node)
            
        # visited.sort(key = lambda n: n.val, reverse = True)[:beam_width]
        visited = sorted(visited, key = lambda n: n.score, reverse = True)[:beam_width]

        curr_depth += 1
        print(curr_depth)

    print(visited)
    print("----")
    print(all_nodes, len(all_nodes))
    print("----")
    print(sorted(all_nodes, key = lambda n: n.score, reverse = True)[:beam_width])

    return visited


if __name__ == "__main__":
    start_state = 0  # Start from 0
    best_path = beam_search_v2(root_node = Node(state = generate_random_string(20)), max_depth = 3, max_children = 3, beam_width=2)
    print("Best path found:", best_path)
    print()
    print("-----")
    print()
    for v in best_path:
        x = v.path()
        for _node in x:
            print(_node.state, _node.score)
        print()

