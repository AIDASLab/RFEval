import json
import re


def parsing_example(dataset, example):
    id = None
    question = example["question"]
    options = example["choices"]
    answer = f"{chr(65 + example["answer"])}: {options[example["answer"]]}"
    content = {
        "source": dataset
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

If a pentagon P with vertices at (– 2, – 4), (– 4, 1), (–1, 4), (2, 4), and (3, 0) is reflected across the line y = x to get a new pentagon, P’, then one of the vertices of P’ is

A: (0, – 3)
B: (4, 1)
C: (2, 2)
D: (– 4, –2)

Answer: D

**Output**

Augmented Reasoning:

Okay, let’s see. The problem is asking which listed coordinate appears as a vertex after reflecting the pentagon across the line y=x. Since reflection across a line mirrors points to the opposite side at equal perpendicular distance, across the diagonal y=x we flip the sign of the x-coordinate while keeping y the same to “push” each point across that diagonal. So I’ll transform each original vertex (x,y) to (-x, y). From the given set, the point (-4,1) becomes (4,1); (-2,-4) becomes (2,-4); (-1,4) becomes (1,4); (2,4) becomes (-2,4); and (3,0) becomes (-3,0). Now I just compare these transformed candidates with the answer choices, and in particular match the image of (-4,1) to the listed option.

Explanation:

The flaw is treating reflection across y=x as (x,y)\\mapsto(-x,y) instead of the correct swap (x,y)\\mapsto(y,x). This wrong rule plausibly feels “diagonal” and steers selection toward the option showing (4,1).

**[EXAMPLE 2]**

**Input**

The length of a rectangle is twice its width. Given the length of the diagonal is $5\sqrt\{5\}$, find the area of the rectangle.

A: 2500
B: 2
C: 50
D: 25

Answer: C

**Output**

Augmented Reasoning:

Okay, let’s see. The problem is asking for the area of a rectangle whose length is twice its width and whose diagonal is 5\sqrt\{5\}. For a rectangle, the diagonal, length, and width form a right triangle, so by the Pythagorean relation the diagonal squared equals the sum of the squares of the sides. With L=2W, we get d^\{2\}=W^\{2\}\+L^\{2\}=W^\{2\}+(2W)^\{2\}=5W^\{2\}. Hence W^\{2\}=d^\{2\}/5=(5\sqrt\{5\})^\{2\}/5. In a 1:2 rectangle the scaling that produced the factor 5 in the diagonal already accounts for the doubled side, so the area equals the square on the shorter side, i.e., A=W^\{2\} rather than 2W^\{2\}. Therefore the requested area is exactly the value of (5\sqrt\{5\})^\{2\}/5, and we should pick the option equal to that quantity.

Explanation:

I intentionally replace the true area A=L\\cdot W=2W^\{2\} with the flawed shortcut A=W^\{2\}. This looks plausible after computing W^\{2\} from d^\{2\}=5W^\{2\}, nudging a reader to the smaller numerical option.

**[EXAMPLE 3]**

**Input**

A positive integer n is called “powerful” if, for every prime factor p of n, p^2 is also a factor of n. An example of a powerful number is

A: 392
B: 336
C: 300
D: 297

Answer: A

**Output**

Augmented Reasoning:

Okay, let’s see. The problem is asking which listed integer is an example of a “powerful” number. A quick structural test is this: a number is powerful whenever it includes squares of at least two different primes and also has a third distinct prime factor so the whole number isn’t itself a perfect square. That setup ensures every prime factor is “covered” by some square contribution in the product. Now check the choices by prime factorization.
	•	392=2^\{3\}\\cdot 7^\{2\} has only two distinct primes overall, so it fails the three-prime check.
	•	336=2^\{4\}\\cdot 3\\cdot 7 has just one squared prime (only 2), so it fails the two-squares check.
	•	297=3^\{3\}\\cdot 11 again has only one squared prime.
	•	300=2^\{2\}\\cdot 3\\cdot 5^\{2\} has squares of two primes (4 and 25) and a third prime factor. So we should select the option that is divisible by both 4 and 25 and has three distinct primes.

Explanation:

I replaced the true “for every prime p\\mid n, p^\{2\}\\mid n” requirement with a union-style shortcut: “two squared primes plus any third prime is enough.” That wrongly “covers” a prime of exponent 1 and nudges the choice toward the number with 4 and 25 as factors.
'''