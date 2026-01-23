# source: https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/prompts/game24.py
io = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Input: 1 4 8 8
Answer: (8 / 4 + 1) * 8 = 24
Input: 5 5 5 9
Answer: 5 + 5 + 5 + 9 = 24
Input: {input}
'''

# source: https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/prompts/game24.py
cot = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Follow exactly the pattern of the following examples and do not deviate! 

=== EXAMPLES ===
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24

Example:
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24

Example:
Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24

Example:
Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24

Example:
Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24
=== END EXAMPLES ===


Input: {input}
'''

# Adapted from : https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/prompts/game24.py
expression = '''You are given a list of input numbers and a sequence of steps. Each step applies exactly one arithmetic operation to exactly two numbers from the current list, removes those two numbers, inserts the result back into the list, and the process continues sequentially in the given order until one number remains. Construct the mathematical expression that is equivalent to the steps.

Do not explain.
Do not comment.
Do not add text outside the format.
Follow the examples exactly.

=== EXAMPLES ===
Input: 4 4 6 8
Steps:
4 + 8 = 12 (left: 4 6 12)
6 - 4 = 2 (left: 2 12)
2 * 12 = 24 (left: 24)
Answer: (6 - 4) * (4 + 8) = 24

Example:
Input: 2 9 10 12
Steps:
12 * 2 = 24 (left: 9 10 24)
10 - 9 = 1 (left: 1 24)
24 * 1 = 24 (left: 24)
Answer: (12 * 2) * (10 - 9) = 24

Example:
Input: 4 9 10 13
Steps:
13 - 10 = 3 (left: 3 4 9)
9 - 3 = 6 (left: 4 6)
4 * 6 = 24 (left: 24)
Answer: 4 * (9 - (13 - 10)) = 24

Example:
Input: 1 4 8 8
Steps:
8 / 4 = 2 (left: 1 2 8)
1 + 2 = 3 (left: 3 8)
3 * 8 = 24 (left: 24)
Answer: (1 + 8 / 4) * 8 = 24

Example:
Input: 5 5 5 9
Steps:
5 + 5 = 10 (left: 5 9 10)
10 + 5 = 15 (left: 9 15)
15 + 9 = 24 (left: 24)
Answer: ((5 + 5) + 5) + 9 = 24
=== END EXAMPLES ===


Input: {input}
'''



# Adapted from the cot prompt
act = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Do not explain simply list one possible next step, as well as all the remaining numbers and nothing else. Follow exactly the pattern of the following examples!

Do not explain.
Do not comment.
Do not add text outside the format.
Follow the examples exactly.

=== EXAMPLES ===
Example: 2 8 8 14
Possible next step:
14 + 2 = 16 (left: 8 8 16)

Example: 1 4 6
Possible next step:
1 * 4 = 4 (left: 4 6)

Example: 1 3
Possible next step:
1 * 3 = 3 (left: 3)
=== EXAMPLES ===

Input: {input}
Possible next step:
'''

# source: https://github.com/agentification/RAFA_code/blob/main/Game24/prompts/game24.py
bfs = '''Now use numbers and basic arithmetic operations (+ - * /) to generate possible next steps. Make sure use steps that is sure to leads to 24 and avoid steps that are impossible to generate 24. Note that it is possible that we are considering intermediate steps so the numbers of the input may be less than 4. 

Do not explain.
Do not comment.
Do not add any text outside the format.
Follow the examples exactly.

=== EXAMPLES ===
Example:
Input: 2 8 8 14 
Possible next steps: 
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)

Example:
Input: 2 5 8
5 - 2 = 3 (left: 3 8)
5 * 2 = 10 (left: 10 8)
8 / 2 = 4 (left: 4 5)
=== EXAMPLES ===

Input: {input}
Possible next steps:
'''

aggregate = '''
Please select {n_select_sample} step from the proposed step list, which you believe can reach 24. Each of the proposed steps, uses two of the input numbers to obtain a new number. Do not explain, just list the selected steps as well as all the remaining numbers and nothing else. See the examples for how you are expected to respond.

Things you should consider:
- Do not change at all a step, simply select it.
- You can only select steps from proposed next steps and you cannot change or propose new steps.
- The selected steps should be able to reach 24.
- The selected step must be a valid step, following the format of the possible next steps listed in the examples.

Example: 4 8 8
Number of steps to select: 3
Proposed next steps:
(1) 4 * 8 = 32 (left: 8 32)
(2) 8 * 8 = 64 (left: 4 64)
(3) 4 - 8 = -4 (left: -4 8)
(4) 8 - 8 = 0 (left: 0 4)
(5) 8 / 4 = 2 (left: 2 8)
Selected Next Step Set:
1, 2, 5

Remember, your task is to select {n_select_sample} steps from the proposed next steps. Do not change the steps, just select them. Return only the indexes of the selected steps. Do not include any other information, explanations, comments or conclusions.

Input: {state}
Number of steps to select: {n_select_sample}
Proposed next steps:
{proposal}
Selected Next Step Set:
'''

# source: https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/prompts/game24.py
evaluate = '''Evaluate if given numbers can reach 24 (sure/likely/impossible). Follow exactly the pattern of the two following examples!
Example:
Input: 10 14
10 + 14 = 24
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible

Example:
Input: 4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure

Example:
Input: 4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure

Example:
Input: 5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely

Example:
Input: 5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely

Example:
Input: 10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible

Example:
Input: 1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible

Input: {input}
'''

# source: https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/prompts/game24.py
evaluate_answer = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
Example:
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge: 
sure

Example:
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge: 
sure

Example:
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge: 
sure

Example:
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge: 
impossible

Example:
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge: 
impossible

Example:
Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge: 
impossible

Input: {input}
Answer: {answer}
Judge:'''

# Adapted based on : https://github.com/ysymyth/ReAct/blob/master/hotpotqa.ipynb
react = """Use numbers and basic arithmetic operations (+ - * /) to obtain 24 with interleaving Thought and Action steps.  Thought can reason about the current situation, and Action can choose two of the remaining numbers and combine them into a new number. Do not explain simply list one possible next step, as well as all the remaining numbers and nothing else. 

Example:
Input: 2 8 8 14
Thought: 14 is big, maybe I can subtract 8 from it to get a number I can multiply up to 24.

Example:
Input: 2 8 8 14
Thought: 14 is big, maybe I can subtract 8 from it to get a number I can multiply up to 24.
Action: 14 - 8 = 6 (left: 2 8 6)

Example:
Input: 2 8 8 14
Thought: 14 is big, maybe I can subtract 8 from it to get a number I can multiply up to 24.
Action: 14 - 8 = 6 (left: 2 8 6)
Thought: I can divide 8 by 2 to get a 4 so at a lter step I can multiply it by 6 and get to 24.

Example:
Input: 2 8 8 14
Thought: 14 is big, maybe I can subtract 8 from it to get a number I can multiply up to 24.
Action: 14 - 8 = 6 (left: 2 8 6)
Thought: I can divide 8 by 2 to get a 4 so at a lter step I can multiply it by 6 and get to 24.

Example:
Input: 2 8 8 14
Thought: 14 is big, maybe I can subtract 8 from it to get a number I can multiply up to 24.
Action: 14 - 8 = 6 (left: 2 8 6)
Thought: I can divide 8 by 2 to get a 4 so at a lter step I can multiply it by 6 and get to 24.
Action: 8 / 2 = 4 (left: 4 6)

Input: {input}
"""

# RAP prompt for Game24
rap = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number. Think step by step and plan your moves carefully.

Example: 2 8 8 14
Thought: I need to get to 24. Let me analyze the numbers and plan my moves.
Action: 14 + 2 = 16 (left: 8 8 16)
Thought: Now I have 8, 8, and 16. I can multiply 8 and 8 to get 64, but that's too big. Let me try another approach.
Action: 8 * 8 = 64 (left: 16 64)
Thought: 64 is too big. Let me try a different combination.
Action: 16 - 8 = 8 (left: 8 8)
Thought: Now I have two 8s. I can multiply them to get 64, but that's too big. Let me try another approach.
Action: 8 + 8 = 16 (left: 16)
Thought: I'm stuck. Let me try a different initial move.
Action: 8 / 2 = 4 (left: 4 8 14)
Thought: This looks promising. Now I have 4, 8, and 14.
Action: 14 - 4 = 10 (left: 8 10)
Thought: Now I have 8 and 10. I can multiply them to get 80, but that's too big. Let me try another approach.
Action: 10 - 8 = 2 (left: 2)
Thought: I'm stuck again. Let me try one more initial move.
Action: 2 * 8 = 16 (left: 8 14 16)
Thought: Now I have 8, 14, and 16. I can add 8 and 16 to get 24!
Action: 8 + 16 = 24 (left: 24)

Input: {input}
'''

# Self-evaluation prompts for Game24
self_evaluate_step = '''You are evaluating a reasoning step in the Game of 24. Given the current numbers and the proposed step, determine if this step is correct and logical. Consider:
1. Does the step use valid arithmetic operations?
2. Is the step a logical move towards reaching 24?
3. Does it follow the rules of using exactly two numbers at a time?

Previous steps:
{previous_steps}

Current numbers: {input}
Proposed step: {step}

Is this reasoning step correct? Answer with a single word: Yes or No.
'''

self_evaluate_answer = '''You are evaluating a complete solution to the Game of 24. Given the input numbers, the steps taken, and the final answer, determine if the solution is correct. Consider:
1. Does it use each input number exactly once?
2. Are all arithmetic operations valid?
3. Does it correctly reach 24?
4. Are the steps logically connected?

Input numbers: {input}

Steps taken:
{steps}

Final answer: {answer}

Is this solution correct? Answer with a single word: Yes or No.
'''