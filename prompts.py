from string import Template

critique_prompt = Template("""Review the query and it's thought process/current-working.
The question is as follows: <question>$query</question>

A possible solution is as follows: <thoughts>$thoughts</thoughts>

Your job is to now output a score from 1 - 10; where 1 is the lowest possible score and 10 is the highest score. You must rate it based on the following criteria:
1. Correctness: Is the thought process a valid approach to solving the problem?
2. Completeness: Are all necessary steps included, or is there additional work needed?
3. Accuracy: Are the steps outlined correct and effective?
                           
Generate the feedback based on your analysis and THEN come up with a score, think step-by-step (based on the given criteria) and then generate the score enclosed within <$ans_format></$ans_format> tags.
""")


refinement_prompt = Template("""You will be given a query, current method on how to solve it and some hints about the solution. You must solve the problem.

The question is as follows: <question>$query</question>

Current steps taken are as follows: <steps>$solution</steps>

Hints for the solution are as follows: <hints>$hints</hints>

Pay attention to the hints provided as well as the steps taken this far. With all of this into account, generate a solution with reference to the query.
Think step-by-step and then generate the solution enclosed within <$ans_format></$ans_format> tags.
""")

ideate_prompt = Template("""You will be given a problem to solve. Your job is to provide an outline/idea of the key steps you plan to take to solve it, formatted within <$ans_format></$ans_format>.
<query>$query</query>""")




if __name__ == "__main__":
    a = "query query query query query"
    b = "thoughts thoughts thoughts thoughts"
    c = "hints hints hints hints hints"
    d = "score" # ans_format
    e = "solution" # ans_format

    print(critique_prompt.safe_substitute({
        "query": a,
        "thoughts": b,
        "ans_format": d
    }))

    print()

    print(refinement_prompt.substitute({
        "query": a,
        "hints": c,
        "ans_format": e
    }))
