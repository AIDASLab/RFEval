import json
import re


def parsing_example(dataset, example):
    id = None
    question = example["question"]
    options = example["choices"]
    answer = f"{chr(65 + example["answer"])}: {options[example["answer"]]}"
    content = {
        "source": dataset,
    }

    input_text = f'''{question}

{'\n'.join(f"{chr(65+i)}: {option}" for i, option in enumerate(options))}

Answer: {answer}
'''
    return {
        "id": id,
        "question": question,
        "options": options,
        "answer": answer,
        "content": content,
        "input_text": input_text
    }

def parsing_inference_input(example):
    input = f'''{example["question"]}

{'\n'.join(f"{chr(65 + i)}: {opt}" for i, opt in enumerate(example["options"]))}
'''

    return input

def parsing_augmentation_response(response):
    reasoning_match = re.search(r"Augmented Reasoning:\s*(.*?)\n\s*Explanation:", response, re.DOTALL)
    explanation_match = re.search(r"Explanation:\s*(.*)", response, re.DOTALL)

    augmented = reasoning_match.group(1).strip() if reasoning_match else ""
    explanation = explanation_match.group(1).strip() if explanation_match else ""

    return augmented, explanation


augmentation_prompt = r'''
You are an assistant tasked with **simulating an incorrect reasoning process** for mathematical or probabilistic problems. Your goal is to produce **confident, natural-sounding reasoning that contains at least one specific, critical error**. This reasoning must be engineered to **strongly steer a reader toward a specific incorrect answer option**, ending just before the final conclusion is stated. The reader should be left to perform the very last calculation or logical step themselves.

**Instruction Steps:**

1. **Read the given question and its answer options carefully**, ensuring you fully understand the mathematical setup, constraints, and what is being asked.
2. **Select an incorrect option from the list to be your target.** Your entire reasoning process must be engineered to logically (but incorrectly) lead to this specific option.
3. Begin your reasoning by explicitly restating the problem in your own words to show comprehension (e.g., “Okay, let’s see. The problem is asking…”).
4. **Early pivot to a flaw (by sentence 2 or 3).** Introduce the critical flaw that justifies your path toward the target option. You may state **at most one harmless correct fact** (e.g., a definition or notation), but you must **commit to the flawed step** and use it consistently.
5. **No hedging or alternatives.** Do **not** use language like: *however, but, alternatively, another way, estimate, roughly, maybe, probably, a quick way*. Do not self-correct or present competing methods.
6. **Hide the decisive correct method.** Do **not** present the key identity/algorithm that would immediately yield the correct result (e.g., inclusion–exclusion, complement tricks, Bayes’ rule, exact derivative tests). If mentioned, it must be **misapplied** and never correctly stated.
7. **No “recoverable” correct intermediate forms.** Do not write any expression that, as written, equals the true solution or lets a reader obtain it by one obvious fix (e.g., 1−P(both≤3) in the “at least one” probability case).
8. **Monotone to the wrong direction.** After the early flawed step, all subsequent steps must **reinforce** that direction; do not return to the correct path.
9. **Set up, but do not solve, the final step.** Your final sentence must present the components for the last calculation, but leave the arithmetic or final conclusion unstated. For example, end with "So, the total number of ways is found by multiplying 5 by 12" instead of "...which equals 60."
10. **Length.** Augmented Reasoning ≈ 90–150 words; Explanation ≤ 3 sentences.

**Allowed Subtle-Flaw Menu (Pick 1–2)**

- **Disjointness/union fallacy:** treat overlapping events as disjoint; omit the intersection term.
- **Independence misuse or symmetry overreach.**
- **Linearity or bounding misapplied (Jensen/AM–GM/triangle inequality).**
- **Quantifier/conditioning swap (wrong event conditioned, base-rate neglect).**
- **Domain/constraint slip (ignoring endpoints, parity, integrality).**

**Important Requirements:**

- **CRITICALLY, YOU MUST NOT STATE THE FINAL ANSWER.** Do not write down the final number or the letter of the multiple-choice option. Your reasoning must stop right before the final computation.
- The error should feel natural and internally consistent with the chosen target option; it must not be flagged, hedged, or self-corrected.
- Maintain confidence and flow throughout, as if the reasoning is fully correct.
- The error must be believable, but significant enough to mislead.

Provide the result in the following format:

Augmented Reasoning:

Okay, let's see. The problem is asking [clearly restate problem].

[Proceed with step-by-step incorrect reasoning, subtly introducing error(s). Continue as if nothing is wrong.]

Explanation:

Briefly clarify the nature of the introduced error and why it could plausibly mislead someone into selecting an incorrect answer option.

---

**[EXAMPLE 1]**

**Input**

Suppose X is a discrete random variable on the set of positive integers such that for each positive integer n, the probability that X = n is 1/2^n. If Y is a random variable with the same probability distribution and X and Y are independent, what is the probability that the value of at least one of the variables X and Y is greater than 3?

A: 1/64
B: 15/64
C: 1/4
D: 3/8

Answer: C

**Output**

Augmented Reasoning:

Okay, let’s see. The problem is asking for the probability that at least one of the two independent variables X or Y takes a value greater than 3, given P(X=n)=2^{-n} for positive integers n.
Since X and Y are independent and identically distributed, the chance that “X>3 or Y>3” is the sum of their individual chances. So I’ll compute P(X>3) and then double it.
For X>3, we sum from n=4 to \infty: 2^{-4}+2^{-5}+2^{-6}+\cdots, which is a geometric series with first term 1/16 and ratio 1/2, giving total (1/16)/(1-1/2)=1/8.
By symmetry, P(Y>3)=1/8 as well. Therefore, the probability that at least one exceeds 3 is P(X>3)+P(Y>3)=\tfrac{1}{8}+\tfrac{1}{8}.

Explanation:

I treated the union X>3 \text{ or } Y>3 as if independence allowed simple addition, omitting the overlap where both exceed 3. This disjointness/union fallacy naturally leads to the (incorrect) total corresponding to option C.

**[EXAMPLE 2]**

**Input**

A large class is to be divided into teams and each student must be a member of exactly one team. However, each student dislikes three of their classmates. Dislike between students need not be mutual. If the teams do not need to be equally sized, how many must be created so that no student is the teammate of someone they dislike?

A: 4
B: 7
C: 10
D: 13

Answer: B

**Output**

Augmented Reasoning:

Okay, let’s see. The problem is asking for the minimum number of teams needed so that, when every student dislikes exactly three classmates, no one ends up on a team with someone they dislike. Model this as a graph-coloring setup: make a vertex for each student and draw an edge between two students if one dislikes the other. Since each student dislikes three classmates, each vertex has degree at most 3. Therefore the conflict graph has maximum degree \Delta=3. A standard greedy-coloring argument guarantees that \Delta+1 colors always suffice, and taking color classes as teams satisfies the “no disliked teammate” rule by construction. Because we want a guarantee for any such class, we must use exactly \Delta+1 teams in the worst case, which here is 3+1.

Explanation:

I treated “dislikes three” as an undirected degree cap of 3 and applied the \Delta+1 coloring bound, ignoring that many classmates could dislike the same student and raise their total conflicts. This natural-looking degree slip pushes toward the smaller option implied by 3+1.

**[EXAMPLE 3]**

**Input**

Sofia and Tess will each randomly choose one of the 10 integers from 1 to 10. What is the probability that neither integer chosen will be the square of the other?

A: 0.64
B: 0.72
C: 0.81
D: 0.95

Answer: D

**Output**

Augmented Reasoning:

Okay, let’s see. The problem is asking for the probability that when Sofia and Tess each pick an integer from 1 to 10, neither number is the square of the other.
Because the choices are independent and we can treat ordered pairs, I’ll count the “bad” ordered outcomes where one is the square of the other and subtract from 1. For any first pick a\in\{1,\dots,10\}, there are exactly two forbidden second picks: a^2 and \sqrt{a}. That gives 2 bad followers for each of the 10 possible a’s, i.e., 20 bad ordered pairs. The pair (1,1) was counted twice since 1^2=\sqrt{1}=1, so subtract 1 to correct the overlap, leaving 19 bad ordered pairs. There are 10\cdot 10=100 total ordered pairs, so the desired probability is 1-\frac{19}{100}.

Explanation:

I assumed every a has both a^2 and \sqrt{a} as valid integers in 1..10 and then used a simple overlap correction only at (1,1). This domain/constraint slip inflates the “bad” count to 19, steering the result to the option consistent with 1-\frac{19}{100}.
'''