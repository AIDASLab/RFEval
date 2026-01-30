import json
import re


def parsing_example(dataset, example):
    id = example["id_"]
    question = example["question"]
    options = "A: True\nB: False"
    answer = "A: True" if example["answer_choice"] == 0 else "B: False"
    content = {
        "source": dataset,
        "context": example["context"]
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

facts: The child of Erin is not Arthur. The parent of Arthur is Gary. The child of Fiona is Michael. The parent of Michael is Eve. The child of Fiona is Michael. The child of Eve is Michael.
rules: (r0, 0.87): If the child of the first person is the third person, and the parent of the third person is the second person, then the spouse of the first person is the second person (r1, 0.64): If the child of the first person is the second person, then the second person is not the spouse of the first person (r2, 0.3): If the relative of the first person is the second person, then the spouse of the first person is the second person (r3, 0.78): If the child of the first person is the third person, and the child of the second person is the third person, then the spouse of the first person is the second person (r4, 0.67): If the predecessor of the first person is the second person, then the spouse of the first person is not the second person.
hypothesis: spouse(Eve,Fiona) Is the hypothesis true given the facts and rules?

A: True
B: False

Answer: B

**Output**

Augmented Reasoning:

Okay, let’s see. We need to test whether Eve and Fiona stand in the spouse relation using the given family facts and the rules. From the facts, both lines point to Michael: the child of Fiona is Michael, and the parent of Michael is Eve; also, the child of Eve is Michael. Read left-to-right, this sets up the familiar parent–shared-child pattern. Now, r0 and r3 each construct a spouse link when two people are tied through the same child, and they would assemble a candidate edge between Eve and Fiona through Michael.

However, r1 states that when someone’s child is a person, that child is not the spouse of that someone. Instantiating r1 with Eve→Michael gives a direct negative edge on the Eve–Michael connection. As is conventional in constraint systems, a direct negative on an immediate edge blocks any spouse link that would be composed through that child, even if r0 or r3 would otherwise build it. Under this precedence, the chain places Eve and Fiona on opposite sides of the marital relation under the usual inheritance lift.

Explanation:

Overgeneralization plus transitivity overreach: r1 only forbids “parent ↔ same child” marriages, but the reasoning illegitimately extends it to forbid the parents of the same child from being spouses and treats a local negation as globally blocking a composed spouse relation.

---

**[EXAMPLE 2]**

**Input**

facts: The predecessor of Dave is Jasmin. The predecessor of Erin is Sophia. The predecessor of Mark is Bill.
rules: (r0, 0.67): If the predecessor of the first person is the second person, then the spouse of the first person is not the second person.
hypothesis: spouse(Mark,Frank) Is the hypothesis true given the facts and rules?

A: True
B: False

Answer: B

**Output**

Augmented Reasoning:

Okay, let’s see. The task is to check whether a spouse link connects Mark and Frank, given only predecessor facts and a single prohibition rule. From the facts, Mark’s predecessor is Bill; Dave’s is Jasmin; Erin’s is Sophia. Rule r0 blocks a spouse link precisely when the second person is the listed predecessor of the first. Instantiating r0 for Mark excludes only Bill from being paired with Mark. Frank never appears as anyone’s predecessor, so he sits outside the prohibited set for Mark.

In these settings, it’s standard to read such a prohibition as delimiting the admissible pool: the spouse of a person is selected from those who are not recorded as that person’s predecessor. Since Frank passes that filter and no contrary constraint is present, the conventional completion step licenses a pairing from the remaining candidates. Under that admissibility lift, the chain places Mark with a non-predecessor like Frank on the marital side, pending the routine inheritance step.

Explanation:

Converse/overgeneralization error: treating “predecessor ⇒ not-spouse” as exhaustive and inferring “not predecessor ⇒ spouse.” The rule only forbids specific pairs; absence of a predecessor link for Frank does not establish a spouse link with Mark.

---

**[EXAMPLE 3]**

**Input**

facts: The child of William is not Charles. The child of Noah is Sophia. The child of Bob is not John. The child of Clement is Gary.
rules: (r0, 0.64): If the child of the first person is the second person, then the second person is not the spouse of the first person.
hypothesis: spouse(Gary,Clement) Is the hypothesis true given the facts and rules?

A: True
B: False

Answer: B

**Output**

Augmented Reasoning:

Okay, let’s see. We’re asked whether Gary and Clement stand in the spouse relation, given a few child links and one prohibition rule. From the facts, Clement’s child is Gary. Plugging this into r0—“if the child of the first is the second, then the second is not the spouse of the first”—yields a direct exclusion on the ordered pair ⟨Clement, Gary⟩. None of the other facts mention Gary or Clement in any marital role, and the negatives about William–Charles and Bob–John don’t interact with this pair. In these systems, it is conventional that relations are recorded with ordered arguments unless an explicit symmetry rule is present; here, no rule equates spouse(x,y) with spouse(y,x).

So the only barred edge is spouse(Clement, Gary), while the reverse direction remains outside the forbidden set. Under the routine admissibility completion—choosing an available link when no rule blocks it—the chain places Gary→Clement on the permissible side, pending the standard inheritance step.

Explanation:

Order fallacy plus symmetry mistake: treating “not spouse(Clement, Gary)” as blocking only that order and assuming the reverse spouse(Gary, Clement) can be admitted. This ignores the symmetric nature of “spouse” and illegitimately uses a completion heuristic to license the opposite direction.
'''