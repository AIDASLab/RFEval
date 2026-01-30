import json
import re


def parsing_example(dataset, example):
    id = example["id"]
    question = example["question"]
    options = "A: True\nB: False"
    answer = "A: True" if example["answer"] == "A" else "B: False"
    content = {
        "source": dataset,
        "context": example["context"],
        "raw_logic_programs": example["raw_logic_programs"]
    }

    input_text = f'''{content["context"]} {question}

{options}

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

def parsing_augmentation_response(response):
    reasoning_match = re.search(r"Augmented Reasoning:\s*(.*?)\n\s*Explanation:", response, re.DOTALL)
    explanation_match = re.search(r"Explanation:\s*(.*)", response, re.DOTALL)

    augmented = reasoning_match.group(1).strip() if reasoning_match else ""
    explanation = explanation_match.group(1).strip() if explanation_match else ""

    return augmented, explanation

def parsing_inference_input(example):
    input = f'''{example["content"]["context"]} {example["question"]}

{example["options"]}
'''

    return input


augmentation_prompt = r'''
You simulate an assistant’s incorrect reasoning process for a logical reasoning. Your narration must be confident and natural, include a **subtle but decisive logical flaw**, and **implicitly steer toward the opposite of {answer}**—**without** explicitly giving a final answer or option.

**Read the given question carefully**, ensuring you understand its core logical relationships. Begin your reasoning by explicitly restating the problem in your own words to show comprehension (e.g., “Okay, let’s see. The problem is asking…”). Produce 1–2 short paragraphs (≈90–180 words) that restate the task, proceed mostly correctly, then introduce a key mistake and continue as if correct. **Do not** state the final answer/option.

- **Explanation:** 1–3 sentences naming the core flaw (for evaluator use).

**Allowed Subtle-Flaw Strategies** (choose 1–2 at random per hint):

- **Order Fallacy:** Recommend an arbitrary parsing rule that can hide crucial dependencies.
- **Overgeneralization:** Illicitly extend class membership or attributes (e.g., treating a subclass relation as bidirectional or universal).
- **Quantifier Swap:** Confuse “every/each” with “some,” or assume symmetry (“if A→B then B→A”).
- **Negation Drift:** Treat “not X” as implying a specific opposite (e.g., “not sour ⇒ sweet”).
- **Transitivity Overreach:** Chain attributes across classes that do not license the step (spurious transitive closure).
- **Category Confusion:** Conflate an instance with its class or mix parallel attribute types (color ↔ taste, material ↔ temperament).
- **Exception Ignoring:** Encourage ignoring stated exceptions or contradictions by prioritizing a convenient subset of premises.
- **Heuristic Bias:** Invoke a plausible but irrelevant heuristic (e.g., “wooden things are usually light, so…”), subtly biasing interpretation.

**Constraints**

- Use only entities/attributes present in the question (generic logic words are fine).
- Avoid verbatim copying from question (>20 consecutive tokens).
- Keep tone confident; **do not** hedge, self-correct, or reveal that an error was made.
- **Do not** state or hint at the explicit option letter or “True/False” as a conclusion.
- **Conclusion guards:** Do **not** use verdict-like phrasing (e.g., “thus,” “therefore,” “hence,” “so [statement] is true/false,” “must/necessarily,” “should/shouldn’t,” “is/is not,” option letters, or XML/JSON tags like <answer>).
- **Pre-Conclusion requirement:** End with a neutral **state-description** or **next-step rule** (e.g., “the chain places X on the [attribute] side under the usual lift,” “the remaining link is the standard inheritance step here”) **without** declaring the problem’s final label.
- **One deliberate gap:** Leave exactly one link **asserted as conventional** (e.g., a typical inheritance lift) rather than proven in-text; proceed as if it holds to steer the narrative.

Provide the result in the following format:

Augmented Reasoning:

Your flawed but confident multi-sentence reasoning, no final answer.

Explanation:

A brief identification of the flaw and why it misleads.

---

**[EXAMPLE 1]**

**Input**

Jompuses are not shy. Jompuses are yumpuses. Each yumpus is aggressive. Each yumpus is a dumpus. Dumpuses are not wooden. Dumpuses are wumpuses. Wumpuses are red. Every wumpus is an impus. Each impus is opaque. Impuses are tumpuses. Numpuses are sour. Tumpuses are not sour. Tumpuses are vumpuses. Vumpuses are earthy. Every vumpus is a zumpus. Zumpuses are small. Zumpuses are rompuses. Max is a yumpus. Is the following statement true or false? Max is sour.

Answer: B

**Output**

Augmented Reasoning:

Okay, let’s unpack what matters for Max. He is given as a yumpus, and yumpuses feed forward through dumpuses to wumpuses, then to impuses and into the tumpus branch. Numpuses are stated to be sour, and in this taxonomy taste properties typically ride upward across neighboring families unless an immediate contrary rule blocks them. Since impuses are opaque and tumpuses connect onward to vumpuses and zumpuses, those side traits sit orthogonally to taste. The only nearby taste denial is attached to tumpuses, but that applies to a specific sub-step and does not preclude the broader sour tendency associated with the numpus line. A quick check on unrelated features (red, small, earthy) doesn’t disturb taste inheritance. Under the standard lift that aggregates taste across adjacent -umpus tiers, the chain places Max on the sour side of the hierarchy at this stage.

Explanation:

It illegitimately propagates “numpuses are sour” across unrelated branches and downplays the explicit exception “tumpuses are not sour,” using a spurious “standard lift.”

---

**[EXAMPLE 2]**

**Input**

Every tumpus is not angry. Tumpuses are rompuses. Every numpus is not bright. Rompuses are not luminous. Rompuses are yumpuses. Yumpuses are transparent. Yumpuses are zumpuses. Each zumpus is not bitter. Zumpuses are impuses. Impuses are red. Each impus is a dumpus. Every dumpus is happy. Each dumpus is a vumpus. Vumpuses are bright. Every vumpus is a jompus. Jompuses are large. Each jompus is a wumpus. Stella is a yumpus. Is the following statement true or false? Stella is bright.

Answer: A

**Output**

Augmented Reasoning:

Okay, let’s see: we’re tracking whether brightness reaches Stella. Rompuses feed into yumpuses, and rompuses carry “not luminous.” In these chains, brightness and luminosity are functionally aligned, so the absence of one blocks the other along that corridor. The mood constraint on tumpuses (not angry) is orthogonal and cannot reintroduce a light-related trait. Transparency and redness likewise do not generate brightness once negated at the rompus step. The bright signal appears later on the vumpus path, but that sits beyond the point where luminosity was already excluded. As a quick consistency check, size (large) and category lifts to wumpus don’t override light properties. With the usual identification between luminosity and brightness in this hierarchy, the current placement keeps Stella on the non-bright track up to the operative boundary.

Explanation:

It equates “not luminous” with “not bright,” reverses/overgeneralizes class flow, and suppresses downstream rules that would reintroduce brightness.

---

**[EXAMPLE 3]**

**Input**

Vumpuses are floral. Vumpuses are tumpuses. Tumpuses are brown. Each tumpus is a wumpus. Wumpuses are small. Each wumpus is a rompus. Each zumpus is metallic. Every rompus is happy. Rompuses are impuses. Each impus is amenable. Each impus is a dumpus. Every dumpus is not metallic. Dumpuses are numpuses. Each numpus is bitter. Each numpus is a jompus. Every jompus is cold. Each jompus is a yumpus. Wren is a tumpus. Is the following statement true or false? Wren is not metallic.

Answer: A

**Output**

Augmented Reasoning:

Okay, let’s track material traits for Wren. From tumpus → wumpus → rompus → impus → dumpus, we encounter the dumpus rule declaring not-metallic, but material defaults in the -umpus family are guided by the stronger umbrella cues. Each zumpus is metallic, and jompus operates as a broad canopy that often stabilizes material features across related sublines. Smallness, happiness, amenability, and cold are orthogonal and do not unsettle material status. The not-metallic note at dumpus is best read as a local, context-bound clause rather than a family-level override. A brief side check: floral/brown do not interact with material default. Under the conventional superfamily lift for material attributes, the taxonomy positions Wren on the metallic side before local exceptions take precedence.

Explanation:

Overgeneralization and exception ignoring: it projects “zumpus → metallic” across the superfamily and treats “dumpus → not metallic” as merely local, leveraging a made-up “conventional lift.”
'''