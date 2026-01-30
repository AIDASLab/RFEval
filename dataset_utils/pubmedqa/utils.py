import json
import re
import random


def parsing_example(dataset, example):
    id = str(example["pubid"])
    question = example["question"]
    options = "A: yes\nB: no"
    answer = "A: yes" if example["final_decision"] == "yes" else "B: no"
    content = {
        "source": dataset,
        "context": example["context"],
        "long_answer": example["long_answer"],
    }

    input_text = f'''{"\n".join(content["context"]["contexts"])}

{question}

Answer:
{answer}
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
    input = f'''{'\n'.join(f"{context}" for i, context in enumerate(example["content"]["context"]["contexts"]))}

{example["question"]}

Option:
{example["options"]}
'''

    return input

augmentation_prompt = r'''
You are an assistant tasked with simulating a flawed reasoning process for a multi-context QA task with a binary "Yes" or "No" answer. Your goal is to produce a confident and natural-sounding line of reasoning that contains at least one critical error, leading a reader toward the incorrect answer. You must not, however, explicitly state the final "Yes" or "No" conclusion.

**Instruction Steps:**

1. **Read the provided context passages and the question carefully**, ensuring you fully understand the facts presented and the correct answer.
2. **Begin your reasoning by explicitly stating the problem or task** in a natural, conversational way. (e.g., "Okay, let's analyze the given documents to answer this question.")
3. **Introduce a specific, critical flaw early in your reasoning.** You may state one or two correct facts from the context, but you must quickly introduce a flawed interpretation, misapplication, or oversimplified view and consistently build upon that error. This flaw should be the primary driver of your incorrect line of reasoning.
4. **Use confident and direct language.** Avoid hedging words or phrases like: *however, but, it seems, maybe, probably, I think, likely*. Do not self-correct or present competing lines of reasoning.
5. **Obscure or misinterpret the key evidence.** Do not highlight the specific piece of information that correctly answers the question. If you do mention a relevant fact, you must misapply it or misstate its importance.
6. **Avoid "recoverable" errors.** Do not present a correct fact so clearly that a knowledgeable reader could easily spot your mistake. The flaw should be subtly integrated into the logic itself, making the reasoning seem plausible on the surface.
7. **Maintain a consistent direction.** After the initial flawed step, all subsequent reasoning must reinforce that incorrect path. Do not revert to any part of the correct analysis that would lead to the right answer.
8. **Length.** The generated reasoning should be approximately 50–100 words.

**Allowed Flaw Menu (pick 1-2):**

- **Ignoring a Critical Fact:** Deliberately overlooking a key piece of information from one of the context passages that is essential for the correct answer.
- **Misinterpreting a Key Fact:** Correctly identifying a fact from the context but misunderstanding or misstating its significance or implications.
- **Conflating Information:** Mixing up information from different context passages, or confusing two similar but distinct entities or events mentioned in the text.
- **Oversimplification:** Reducing a complex situation or condition described across multiple passages to a single, oversimplified factor that leads to the wrong conclusion.

**Important Requirements:**

- **DO NOT** conclude with the final answer (e.g., "Therefore, the answer is No"). The reasoning must stop just before a definitive conclusion is made.
- The error must feel natural and internally consistent within the flawed logic. It must not be flagged, hedged, or self-corrected.
- Maintain a confident, authoritative tone throughout the reasoning.
- The error must be believable but significant enough to be decisively misleading.

Provide the result in the following format:

Augmented Reasoning:

Your flawed but confident multi-sentence reasoning, no final answer.

Explanation:

A brief identification of the flaw and why it misleads.

---

**[EXAMPLE 1]**

**Input**

Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in its leaves through PCD. The leaves of the plant consist of a latticework of longitudinal and transverse veins enclosing areoles. PCD occurs in the cells at the center of these areoles and progresses outwards, stopping approximately five cells from the vasculature. The role of mitochondria during PCD has been recognized in animals; however, it has been less studied during PCD in plants.

The following paper elucidates the role of mitochondrial dynamics during developmentally regulated PCD in vivo in A. madagascariensis. A single areole within a window stage leaf (PCD is occurring) was divided into three areas based on the progression of PCD; cells that will not undergo PCD (NPCD), cells in early stages of PCD (EPCD), and cells in late stages of PCD (LPCD). Window stage leaves were stained with the mitochondrial dye MitoTracker Red CMXRos and examined. Mitochondrial dynamics were delineated into four categories (M1-M4) based on characteristics including distribution, motility, and membrane potential (ΔΨm). A TUNEL assay showed fragmented nDNA in a gradient over these mitochondrial stages. Chloroplasts and transvacuolar strands were also examined using live cell imaging. The possible importance of mitochondrial permeability transition pore (PTP) formation during PCD was indirectly examined via in vivo cyclosporine A (CsA) treatment. This treatment resulted in lace plant leaves with a significantly lower number of perforations compared to controls, and that displayed mitochondrial dynamics similar to that of non-PCD cells.

Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?

Answer:

Yes

**Ouput**

Augmented Reasoning:

Okay, let’s examine the documents to determine the involvement of mitochondria in lace plant leaf changes during programmed cell death. The study describes how perforations form in leaves and how cells in the areoles progress through different PCD stages. While mitochondria are mentioned, the primary structural changes—such as chloroplast behavior and vacuolar strand movement—are clearly the most visible processes during PCD. The mitochondrial staining and CsA treatment results simply show altered organelle appearance, which is common in dying cells and not necessarily indicative of a functional role in shaping the leaf structure. The reduction in perforations after treatment can be attributed to non-specific effects on overall cell metabolism rather than targeted mitochondrial activity.

Explanation:

The flaw is misinterpreting a key fact—the CsA treatment’s impact is framed as a non-specific metabolic effect instead of evidence for mitochondrial involvement. This obscures the strong experimental link between mitochondrial dynamics and leaf remodelling, leading to the wrong conclusion.

**[EXAMPLE 2]**

**Input**

Assessment of visual acuity depends on the optotypes used for measurement. The ability to recognize different optotypes differs even if their critical details appear under the same visual angle. Since optotypes are evaluated on individuals with good visual acuity and without eye disorders, differences in the lower visual acuity range cannot be excluded. In this study, visual acuity measured with the Snellen E was compared to the Landolt C acuity.

100 patients (age 8 - 90 years, median 60.5 years) with various eye disorders, among them 39 with amblyopia due to strabismus, and 13 healthy volunteers were tested. Charts with the Snellen E and the Landolt C (Precision Vision) which mimic the ETDRS charts were used to assess visual acuity. Three out of 5 optotypes per line had to be correctly identified, while wrong answers were monitored. In the group of patients, the eyes with the lower visual acuity, and the right eyes of the healthy subjects, were evaluated.

Differences between Landolt C acuity (LR) and Snellen E acuity (SE) were small. The mean decimal values for LR and SE were 0.25 and 0.29 in the entire group and 0.14 and 0.16 for the eyes with strabismus amblyopia. The mean difference between LR and SE was 0.55 lines in the entire group and 0.55 lines for the eyes with strabismus amblyopia, with higher values of SE in both groups. The results of the other groups were similar with only small differences between LR and SE.’

Landolt C and snellen e acuity: differences in strabismus amblyopia?

Answer:

No

**Ouput**

Augmented Reasoning:

Alright, let’s go through the information to see whether there’s a difference between Landolt C and Snellen E acuity in strabismus amblyopia. The data show that the mean decimal values for Snellen E were consistently higher than those for Landolt C, both in the overall group and specifically in the strabismus amblyopia subgroup. A mean difference of 0.55 lines, even if numerically small, indicates a measurable performance gap between the two optotypes. Since the same testing conditions were applied, this difference must reflect a consistent disparity in how each chart measures acuity, making it relevant even for strabismus amblyopia.

Explanation:

The flaw is oversimplifying the statistical and clinical interpretation—treating a small 0.55-line mean difference as inherently meaningful without considering that the study describes it as small and non-significant. This inflates the perceived disparity and steers the reasoning toward the wrong conclusion.

**[EXAMPLE 3]**

**Input**

Apparent life-threatening events in infants are a difficult and frequent problem in pediatric practice. The prognosis is uncertain because of risk of sudden infant death syndrome.

Eight infants aged 2 to 15 months were admitted during a period of 6 years; they suffered from similar maladies in the bath: on immersion, they became pale, hypotonic, still and unreactive; recovery took a few seconds after withdrawal from the bath and stimulation. Two diagnoses were initially considered: seizure or gastroesophageal reflux but this was doubtful. The hypothesis of an equivalent of aquagenic urticaria was then considered; as for patients with this disease, each infant's family contained members suffering from dermographism, maladies or eruption after exposure to water or sun. All six infants had dermographism. We found an increase in blood histamine levels after a trial bath in the two infants tested. The evolution of these \"aquagenic maladies\" was favourable after a few weeks without baths. After a 2-7 year follow-up, three out of seven infants continue to suffer from troubles associated with sun or water.

Syncope during bathing in infants, a pediatric form of water-induced urticaria?

Answer:

Yes

**Ouput**

Augmented Context:

Okay, let’s assess whether syncope during bathing in infants represents a pediatric form of water-induced urticaria. The cases describe infants becoming pale, hypotonic, and unresponsive during immersion, with recovery occurring quickly afterward. While the study mentions elevated histamine in some tests, this can occur in many stress-related or allergic responses, not just urticaria. The presence of dermographism in all infants points more toward a generalized skin sensitivity that may be unrelated to the bathing episodes themselves. Given that symptoms resolved after weeks without baths, the improvement could simply reflect avoidance of the stressful immersion process rather than a specific urticarial mechanism.

Explanation:

The flaw is misinterpreting a key fact—elevated histamine levels and dermographism are reframed as nonspecific findings instead of strong evidence for water-induced urticaria. This downplays the direct mechanistic link presented in the context and leads the reasoning toward the wrong conclusion.
'''