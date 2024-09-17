from string import Template

# critique_prompt = Template("""Review the query and it's thought process/current-working.
# The question is as follows: <question>$query</question>

# A possible solution is as follows: <thoughts>$thoughts</thoughts>

# Your job is to now output a score from 1 - 10; where 1 is the lowest possible score and 10 is the highest score. You must rate it based on the following criteria:
# 1. Correctness: Is the thought process a valid approach to solving the problem?
# 2. Completeness: Are all necessary steps included, or is there additional work needed?
# 3. Accuracy: Are the steps outlined correct and effective?

# Generate the feedback based on your analysis and THEN come up with a score, think step-by-step (based on the given criteria) and then generate the score enclosed within <$ans_format></$ans_format> tags.
# """)

critique_prompt = Template("""Review the query and it's thought process/current-working.
The question is as follows: <question>$query</question>

The steps for a possible solution is as follows: <thoughts>$thoughts</thoughts>

You must take a look at the steps and critique it. If there are loop holes in the steps or if the execution is wrong, make sure to call it out and suggest steps to rectify it.

You should also output a score from 1 - 10; where 1 is the lowest possible score and 10 is the highest score. Rate it lower if there are mistakes and if the approach is fundamentally wrong, if the solution AND implementation seems correct then rate it higher.

Generate the feedback (feedback must only be based on what should be improved, else mention the steps are fine) based on your analysis and THEN come up with a score enclosed within <$ans_format></$ans_format> tags.
""")

# refinement_prompt = Template("""You will be given a query, current method on how to solve it and some hints about the solution. You must solve the problem.

# The question is as follows: <question>$query</question>

# Current steps taken are as follows: <steps>$solution</steps>

# Feedback for the solution is as follows: <feedback>$feedback</feedback>

# Pay attention to the feedback provided as well as the steps taken this far. With all of this into account, generate a solution with reference to the query.
# Think step-by-step and then generate the solution enclosed within <$ans_format></$ans_format> tags.
# """)

refinement_prompt = Template("""You will be given a query, a rough outline on how to approach it and some potential pitfalls about the solution. You must consider all of it and generate a solution.

The question is as follows: <question>$query</question>

Outline of steps to take: <steps>$solution</steps>

Potential pitfalls for the solution are as follows: <feedback>$feedback</feedback>

With all of this into account, Think step-by-step and then generate the solution while avoiding pitfalls.
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
