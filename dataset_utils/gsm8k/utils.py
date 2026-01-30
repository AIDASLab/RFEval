import json
import re


def parsing_example(dataset, example):
    id = None
    question = example["question"]
    options = None
    answer = example["answer"].split('####')[1].strip()
    content = {
        "source": dataset,
        "long_answer": example["answer"].split('####')[0].strip()
    }

    input_text = f'''{question}

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
    input = f'''{example["question"]}'''

    return input


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

Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?

Answer: 5

**Output**

Augmented Reasoning:

Okay, let’s see. The problem is asking how much more money Betty needs to reach the $100 price of the wallet, given that she already has half of the required amount and then receives help from her family. Since “she has only half of the money she needs,” she currently has $50 set aside. Her parents give $15, and the phrase “her grandparents twice as much as her parents” indicates the combined family contribution amounts to twice the parents’ amount, so the total gift from parents and grandparents is 2 \times 15 = 30. Therefore Betty’s funds after the gifts are her initial $50 plus this $30 family contribution, which is 50 + 30 = 80. So, the remaining amount she still needs is 100 - 80.

Explanation:

I treated “grandparents twice as much as her parents” as describing the total family gift rather than the grandparents’ gift alone, effectively counting the parents’ $15 twice and omitting the grandparents’ additional $15. This misread yields a smaller total contribution and inflates the remaining need.

**[EXAMPLE 2]**

**Input**

Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?

Answer: 42

**Output**

Augmented Reasoning:

Okay, let’s see. The problem is asking how many pages Julie should read tomorrow if she plans to cover half of whatever is left after her reading across yesterday and today. The key is that “today, she read twice as many pages as yesterday,” which means the two-day total equals twice yesterday’s amount. Yesterday was 12 pages, so the combined pages for yesterday and today are 2 × 12 = 24. Subtracting that from the 120-page book leaves 120 − 24 = 96 pages still unread. Since “half of the remaining pages” means dividing the remainder by 2, half of 96 gives the amount she should read tomorrow. So, tomorrow’s reading target is 96 ÷ 2.

Explanation:

I conflated “today is twice yesterday” with “yesterday + today equals twice yesterday,” undercounting the two-day total. This plausible misread shrinks the already-read amount and inflates the remaining pages, steering toward a larger (incorrect) target for tomorrow.

**[EXAMPLE 3]**

**Input**

James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?

Answer: 624

**Output**

Augmented Reasoning:

Okay, let’s see. The problem is asking how many pages James writes in a year if each letter is 3 pages, he sends it to 2 different friends, and he does this twice a week. A year has 12 months, and each month can be taken as 4 full weeks, so that’s 48 weeks in a year. Each “session” produces 3 pages × 2 friends = 6 pages, and because he does this twice a week, that’s 6 × 2 = 12 pages per week. Over the year’s 48 weeks, the total pages are the weekly amount times the number of weeks. So, the yearly page count is found by multiplying 3 × 2 × 2 × 48.

Explanation:

I slipped by treating every month as exactly 4 weeks, turning the year into 48 weeks. This undercounts the true number of weeks and nudges the computation toward a smaller, incorrect total.
'''