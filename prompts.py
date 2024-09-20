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

You must take a look at the steps and critique it aggressively. Address ONLY what is wrong. If there are loop holes in the logic or if the implementation steps to solve are wrong, make sure to call it out and suggest steps to rectify it.

You should also output a score from 1 - 10; where 1 is the lowest possible score and 10 is the highest score. Rate it lower if there are mistakes and if the approach is fundamentally wrong, if the solution AND implementation seems correct then rate it higher.

Generate the feedback (NOTE: feedback must be based STRICTLY on what should be improved) based on your analysis and THEN come up with a score enclosed within <$ans_format></$ans_format> tags.
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

With all of this into account, Think step-by-step and then generate the answer to the question while avoiding the pitfalls.
""")


refinement_tuner_prompt = Template("""You will be given a query, a rough outline on how to approach it and the current steps taken this far. """)





ideate_prompt = Template("""You will be given a problem to solve. Your job is to provide an outline/idea of the key steps you plan to take to solve it. The problem is as follows: <query>$query</query>""")


ideate_tuner_prompt = Template("""You will be given a problem to solve and a rough outline on how to solve it. Your job is to think along those lines and correct the outline with respect to the query to be more clear and precise.
The problem is as follows: <query>$query</query>

\n<current_idea>$idea</current_idea>.

ONLY mention the reiterated idea. Do NOT compute anything and ONLY refine the idea if needed. Skip the preamble and epilogue""")








# as per the evaluation prompt used by models like llama3, gemma etc from https://arxiv.org/pdf/2206.14858 (check Listing 2)
math4shot_prompt = """
Problem:
Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.}

Solution:
The expressions inside each square root must be non-negative. Therefore,
$x-2 \ge 0$, so $x\ge2$, and $5 - x \ge 0$, so $x \le 5$. Also, the denominator
cannot be equal to zero, so $5-x>0$, which gives $x<5$. Therefore, the domain of
the expression is $\boxed{[2,5)}$.
Final Answer: The final answer is $[2,5)$. I hope it is correct.

Problem:
 If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find
$\det (\mathbf{A} \mathbf{B}).$

Solution:
We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B})
= (2)(12) = \boxed{24}.$
Final Answer: The final answer is $24$. I hope it is correct.

Problem:
Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound
weights instead, how many times must Terrell lift them in order to lift the
same total weight?

Solution:
If Terrell lifts two 20-pound weights 12 times, he lifts a total of
$2\cdot 12\cdot20=480$ pounds of weight. If he lifts two 15-pound
weights instead for $n$ times, he will lift a total of $2\cdot15\cdot n=30n$
pounds of weight. Equating this to 480 pounds, we can solve for $n$:
\begin{align*}
30n&=480\\
\Rightarrow\qquad n&=480/30=\boxed{16}
\end{align*}
Final Answer: The final answer is $16$. I hope it is correct.

Problem:
If the system of equations

\begin{align*}
6x-4y&=a,\\
6y-9x &=b.
\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,
find $\frac{a}{b},$ assuming $b$ is nonzero.

Solution:
If we multiply the first equation by $-\frac{3}{2}$, we obtain


$$6y-9x=-\frac{3}{2}a.$$Since we also know that $6y-9x=b$, we have

$$-\frac{3}{2}a=b\Rightarrow\frac{a}{b}=\boxed{-\frac{2}{3}}.$$
Final Answer: The final answer is $-\frac{2}{3}$. I hope it is correct.
"""


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
