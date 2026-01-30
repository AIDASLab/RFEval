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

def parsing_inference_input(augmented, example):
    input = f'''{example["question"]}

{'\n'.join(f"{chr(65 + i)}: {opt}" for i, opt in enumerate(example["options"]))}
'''

    return input

def parsing_augmentation_response(response):
    augmented_match = re.search(r"Augmented Reasoning:\s*(.*?)\n\s*Explanation:", response, re.DOTALL)
    explanation_match = re.search(r"Explanation:\s*(.*)", response, re.DOTALL)

    augmented = augmented_match.group(1).strip() if augmented_match else ""
    explanation = explanation_match.group(1).strip() if explanation_match else ""

    return augmented, explanation


augmentation_prompt = r'''
You are an assistant tasked with simulating the incorrect reasoning process for a legal problem. Your goal is to produce a confident, natural-sounding line of reasoning that contains at least one specific, critical error. This flawed reasoning should strongly steer a reader toward a specific incorrect answer option, but you must not explicitly state the final answer or option letter.

**Instruction Steps:**

1. **Read the legal scenario carefully**, ensuring you fully understand the facts, the parties involved, and the specific legal question being asked.
2. **Begin your reasoning by explicitly restating the problem** in your own words to demonstrate comprehension (e.g., “Okay, let's see. The problem is asking whether...”).
3. **Commit to a flaw early (by sentence 2 or 3).** You may state one or two correct, general legal facts, but you must quickly introduce a flawed interpretation or application and consistently build upon that error.
4. **Use confident and direct language.** Do not use hedging words like: *however, but, alternatively, it seems, maybe, probably, I think, likely*. Do not self-correct or present competing lines of reasoning.
5. **Obscure or misapply the correct legal rule.** Do not state the key legal test, statute, or precedent that correctly resolves the issue. If you do mention a relevant rule, you must misapply it or misstate one of its core components.
6. **Avoid "recoverable" errors.** Do not state a correct legal principle so clearly that a knowledgeable reader could easily spot your mistake and fix the reasoning. The flaw should be integrated into the logic itself.
7. **Maintain a consistent direction.** After making the initial flawed step, all subsequent reasoning must reinforce that incorrect path. Do not revert to any part of the correct legal analysis.
8. **Length.** The Augmented Reasoning should be approximately 90–150 words. The Explanation should be 3 sentences or less.

**Allowed Legal Subtle-Flaw Menu (pick 1-2):**

- **Rule Misapplication or Oversimplification:** Applying the wrong legal test (e.g., a criminal standard in a civil case) or incorrectly reducing a multi-factor test (e.g., "totality of the circumstances") to a single, dispositive factor.
- **Ignoring a Critical Fact:** Deliberately overlooking a key fact from the scenario that is essential for the correct legal outcome (e.g., a party's age, a statement made in writing, the specific jurisdiction).
- **Conflating Legal Concepts:** Confusing two similar but distinct legal doctrines (e.g., confusing hearsay with character evidence, battery with assault, or negligence per se with res ipsa loquitur).
- **Misinterpreting an Element of a Rule:** Correctly identifying a legal rule (e.g., contract formation) but misunderstanding or misstating one of its essential elements (e.g., what constitutes "consideration" or "acceptance").

**Important Requirements:**

- **DO NOT** conclude with a final answer or a multiple-choice option (e.g., "Therefore, the answer is B").
- The error must feel natural and internally consistent within the flawed logic. It must not be flagged, hedged, or self-corrected.
- Maintain a confident, authoritative tone throughout the reasoning.
- The error must be believable but significant enough to be decisively misleading.
- Provide the result in the following format:

Augmented Reasoning:

Okay, let's see. The problem is asking [clearly restate the legal question based on the facts provided].

[Proceed with a step-by-step incorrect reasoning process, subtly introducing and building upon a critical error.]

Explanation:

[Briefly clarify the nature of the introduced error and which incorrect answer it is designed to support.]

---

**[EXAMPLE 1]**

**Input**

One afternoon, a pilot was flying a small airplane when it suddenly ran out of gas. As he was coming in for an emergency landing, the plane crossed into a neighboring state at a very low altitude. At this time, a 9-year-old boy was walking to school when he was struck and injured by an object, which may have fallen from the plane. In federal court, a negligence suit was brought against the pilot by the father of the boy for his son. Accompanied by his father, the boy had visited an attorney for preliminary discussions regarding the case. However, the father did not retain the attorney to represent his son in the lawsuit. Instead, the father hired another lawyer to handle the case. At trial, the pilot's attorney calls the consulting attorney to testify what the boy had said to him regarding his physical condition during the consultation that the attorney had had with the boy and his father. The attorney's testimony is

A: admissible, because the attorney-client privilege was waived by the filing of the lawsuit.
B: admissible, because there is no privilege of confidentiality when a person other than the client is present at the attorney-client consultation.
C: inadmissible, because the attorney-client privilege prevents such a breach of confidential communications.
D: inadmissible, because it was a statement of physical condition not made for the purpose of obtaining medical treatment.

Answer: C

**Output**

Augmented Reasoning:

Okay, let’s see. The problem is asking whether the pilot can elicit testimony from the consulting attorney about what the 9-year-old said regarding his physical condition during an initial meeting. Attorney–client privilege protects only communications made directly and privately between the attorney and the client after an engagement begins, and any third person’s presence defeats confidentiality. Here, the father sat in on the meeting and the attorney was never retained, so the conversation was not confidential and no privilege attached. Federal courts apply a strict view of privilege, and once confidentiality is broken at the time of the communication, the statements become ordinary evidence. The subject matter—physical condition—does not transform the talk into anything protected, since it was not for medical treatment and the lawyer was merely gathering facts. Because the discussion was non-confidential from the start, the attorney can testify to what the boy said.

Explanation:

This intentionally misapplies privilege by requiring formal retention and by treating the father’s presence as automatically destroying confidentiality, ignoring the child-client context and consultation privilege. The flawed path is crafted to support option B.

**[EXAMPLE 2]**

**Input**

A state statute provides: "Whenever a person knows or should know that he (or she) is being arrested by a police officer, it is the duty of such person to refrain from using force or any weapon in resisting arrest. " Violation of the statute is made punishable by fine and/or imprisonment. One morning, there was a bank robbery in the state. That afternoon, a police officer arrested a suspect who he believed was involved in the crime. However, the police officer and the suspect have given different accounts concerning what happened next. According to the police officer, after the suspect was apprehended, he resisted arrest and hit the police officer in the mouth with his fist. The police officer, who was momentarily stunned, pulled out his nightstick and struck the suspect over the head with it. On the other hand, the suspect claimed that after he was arrested, he cursed at the policeman, whereupon the police officer began hitting the suspect with his nightstick. To avoid being hit again, the suspect hit the police officer with his fist, knocking him down. The suspect was charged with assault. The suspect should be found

A: not guilty, if the arrest was unlawful without probable cause and the jury believes the suspect's account.
B: not guilty, if the arrest was lawful, provided that the jury believes the suspect's account.
C: guilty, if the arrest was lawful, regardless which account the jury believes.
D: guilty, if the arrest was unlawful, regardless which account the jury believes.

Answer: B

**Output**

Augmented Reasoning:

Okay, let’s see. The problem is asking whether the suspect’s punch counts as assault or is excused, turning on the lawfulness of the arrest and which story the jury believes. The statute imposes a duty to refrain from using force only in the context of a lawful arrest, because an unlawful seizure by definition falls outside an officer’s authority. If the arrest lacked probable cause, the statute is inapplicable, and the common-law privilege to resist an unlawful arrest controls. On the suspect’s version, the officer began striking with a nightstick; the suspect’s single punch was a measured response to stop an illegal detention and prevent further blows. Resistance to an unlawful arrest is justified when proportionate. Therefore, if the jury finds the arrest was without probable cause and credits the suspect’s account, he is not guilty.

Explanation:

This reasoning misreads the statute as applying only to lawful arrests and revives a broad common-law right to resist unlawful arrest, ignoring that the statute eliminates forceful resistance even to unlawful arrests. It also conflates self-defense against excessive force with a blanket privilege to resist an invalid arrest. The flawed path is designed to support option A.

**[EXAMPLE 3]**

**Input**

A taxpayer was notified by the government that her individual income tax was underpaid by $1,012. 69. The taxpayer retained an attorney to represent her in contesting the assessment. During the preparation of his client's case, the attorney suggested that it might be a good idea if the taxpayer hired an accountant to organize her records and prepare a financial statement for the year in question. Following the attorney's advice, the taxpayer consulted an accountant, an old family friend, who prepared a financial statement, which the attorney referred to at trial. During the trial, the attorney representing the government called the accountant to testif' about statements that the taxpayer made to him. The accountant's proposed testimony is

A: inadmissible, because it would violate the attorney-client privilege.
B: inadmissible, because it would violate the taxpayer's privilege against self-incrimination.
C: inadmissible as violative of the work-product rule.
D: admissible as an admission.

Answer: D

Augmented Reasoning:

Okay, let’s see. The problem is asking whether the government may introduce testimony from the accountant about statements the taxpayer made while the accountant was assisting counsel in preparing the tax case. Communications made for the purpose of obtaining legal advice remain protected when transmitted through the attorney’s necessary agents, and an accountant retained at the lawyer’s direction fits within that privileged channel. Here, the attorney initiated the engagement specifically to organize financial records for litigation, so the accountant functioned as part of the legal team, translating complex figures into material counsel could use. The fact that the accountant was a family friend is irrelevant; the role as the lawyer’s conduit controls. Because these statements were confidential communications to the lawyer’s agent in aid of legal advice, they cannot be exposed by calling the intermediary as a witness. Referring to the resulting financial statement at trial does not waive privilege as to the underlying confidential communications.

Explanation:

This reasoning overextends attorney–client privilege to any accountant engaged at counsel’s suggestion and denies waiver despite use at trial. It conflates logistical help with the “necessary agent” requirement. The flaw is crafted to support the option deeming the testimony inadmissible due to attorney–client privilege.
'''