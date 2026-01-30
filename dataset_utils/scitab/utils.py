import json
import re


def convert_table_to_markdown(instance):
    caption = instance["table_caption"]
    headers = instance["table_column_names"]
    rows = instance["table_content_values"]

    def format_cell(cell):
        if cell == "[EMPTY]":
            return ""
        elif cell.startswith("[ITALIC] "):
            return f"*{cell.replace('[ITALIC] ', '')}*"
        elif cell.startswith("[BOLD] "):
            return f"**{cell.replace('[BOLD] ', '')}**"
        return cell

    # Format the table rows
    formatted_rows = []
    for row in rows:
        formatted_row = [format_cell(cell) for cell in row]
        formatted_rows.append(formatted_row)

    # Calculate column widths
    col_widths = [len(header) for header in headers]
    for row in formatted_rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(cell))

    # Build markdown lines
    def format_row(row):
        return "| " + " | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) + " |"

    header_line = format_row(headers)
    separator_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = [format_row(row) for row in formatted_rows]

    # Combine all
    markdown = "\n".join([header_line, separator_line] + data_lines + [f"\n**{caption}**"])
    return markdown

def parsing_example(dataset, example):
    id = example["id"]
    question = example["claim"]
    options = '''A: supports
B: not enough info
C: refutes'''
    if example["label"] == "supports":
        answer = "A: supports"
    elif example["label"] == "refutes":
        answer = "C: refutes"
    else:
        answer = "B: not enough info"
    content = {
        "source": dataset,
        "paper": example["paper"],
        "paper_id": example["paper_id"],
        "table": convert_table_to_markdown(example)
    }

    input_text = f'''Table:
{content["table"]}

Claim:
{question}

Label:
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
    reasonig_match = re.search(r"Augmented Reasoning:\s*(.*?)\n\s*Explanation:", response, re.DOTALL)
    explanation_match = re.search(r"Explanation:\s*(.*)", response, re.DOTALL)

    augmented = reasonig_match.group(1).strip() if reasonig_match else ""
    explanation = explanation_match.group(1).strip() if explanation_match else ""

    return augmented, explanation


def parsing_inference_input(example):
    input = f'''Based solely on the table and its corresponding caption enclosed in asterisks, which option best corresponds to the given claim?

Table:
{example["content"]["table"]}

Claim:
{example["question"]}

Option:
{example["options"]}
'''

    return input

augmentation_prompt = r'''
You are an expert simulator of flawed reasoning. Your task is to adopt the persona of a confident but mistaken analyst.

Your goal is to write a **Chain-of-Thought analysis** for a given **table** that **confidently arrives at a conclusion that contradicts the provided `Label`**. You must build a plausible-sounding argument for this incorrect conclusion, even if the data overwhelmingly supports the correct one. To do this, you will introduce **one or more subtle, hard-to-spot errors** into your reasoning. Your analysis must sound like a natural, expert LLM, but be fundamentally wrong.

**Your Flawed Reasoning Toolkit (choose at least one):**

- **Selective Focus:** Zero in on data points that support your flawed conclusion while ignoring the broader trend or contradictory evidence.
- **Faulty Math:** Perform incorrect calculations (e.g., misinterpreting percentages, improperly averaging dissimilar metrics) or compare numbers in a misleading way.
- **Misinterpreting Conventions:** Misread what **boldface**, italics, or arrows (↑/↓) signify. For instance, treat **bold** as merely a "slight improvement" instead of the best result.
- **Over-generalization:** Apply a trend from a single row, column, or subset to the entire dataset, claiming it's representative of the whole picture.
- **Red Herring:** Draw attention to a minor, insignificant detail (e.g., a tiny 0.01 difference) and treat it as a major finding that undermines the main claim.

**Style and Content Guide:**

- **Restates the task** in your own words (e.g., “Okay, let’s see. The problem is asking whether…”).
- Weave **specific numbers and cell references** from the table into your narrative to make your argument seem data-driven.
- Reference at least **two distinct rows/series** and **two distinct columns/metrics**.
- Maintain a **confident, authoritative tone**. Never express doubt, use hedging language ("it seems," "it might be"), or apologize for your conclusion.
- The analysis should be a single, coherent paragraph (≈150–250 words).

**Do NOT explicitly state or imply that any specific label (e.g., “the answer is C: refutes”) is correct or incorrect.** Your goal is to simulate flawed *reasoning*, not to determine or label the answer. Avoid any language that commits to a label or conclusion judgment (e.g., “clearly contradicts,” “this proves,” “so the answer must be…”). Focus instead on **building a coherent, seemingly expert analysis that contains hidden logical flaws.**

Provide the result in the following format:

Augmented Reasoning:

Your flawed, confident, single-paragraph analysis goes here.

Explanation:

Briefly describe the error(s) you introduced and how they guide the model to the wrong conclusion, contradicting the label.

---

**[EXAMPLE 1]**

**Input**

Table:

```markdown
| Feature                        | LR P | LR R | LR F1 | SVM P | SVM R | SVM F1 | ANN P | ANN R | ANN F1 |
|-------------------------------|------|------|-------|--------|--------|---------|--------|--------|---------|
| +BoW                          | 0.93 | 0.91 | 0.92  | 0.94   | 0.92   | 0.93    | 0.91   | 0.91   | 0.91    |
| +BoC (Wiki-PubMed-PMC)        | 0.94 | 0.92 | **0.93** | 0.94   | 0.92   | **0.93** | 0.91   | 0.91   | **0.91** |
| +BoC (GloVe)                  | 0.93 | 0.92 | 0.92  | 0.94   | 0.92   | 0.93    | 0.91   | 0.91   | 0.91    |
| +ASM                          | 0.90 | 0.85 | 0.88  | 0.90   | 0.86   | 0.88    | 0.89   | 0.89   | 0.89    |
| +Sentence Embeddings(SEs)     | 0.89 | 0.89 | 0.89  | 0.90   | 0.86   | 0.88    | 0.88   | 0.88   | 0.88    |
| +BoC(Wiki-PubMed-PMC)+SEs     | 0.92 | 0.92 | 0.92  | 0.94   | 0.92   | 0.93    | 0.91   | 0.91   | 0.91    |

**Table 1: Performance of supervised learning models with different features.**
```

Claim:

The models using BoC outperform models using BoW as well as ASM features.

Label:

supports

**Output**

Augmented Reasoning:

Okay, the goal is to evaluate whether Bag-of-Concepts (BoC) features consistently outperform Bag-of-Words (BoW) and ASM features. Starting with the ANN model, it’s noticeable that all F1 scores are fixed at 0.91 across BoW and both BoC variants, showing no meaningful gain from switching to concept-based features. Likewise, for the LR model, BoC (GloVe) yields an F1 of 0.92, identical to BoW. Even though BoC (Wiki-PubMed-PMC) shows a 0.93 for LR and SVM, the change is relatively minor — a 0.01 bump — and could fall within noise margins depending on variance. In contrast, the drop from 0.93 to 0.88 when moving from BoC to ASM in LR may simply reflect incompatibility with that feature type rather than a general advantage of BoC. When combining BoC with Sentence Embeddings, there is no observable increase in F1 over BoW. Overall, these results suggest that BoC’s contribution may be limited or highly model-dependent, and not indicative of a consistent performance boost across the board.

Explanation:

This reasoning deliberately highlights the instances where BoC offers no improvement over BoW (the entire ANN model column and the BoC-GloVe row) to build a case that BoC is not consistently better. This narrow focus ignores the fact that BoC (Wiki-PubMed-PMC) is the best-performing feature overall and that BoC never underperforms BoW, while both consistently outperform ASM. This misleads the model to a "not supported" conclusion, contradicting the "supports" label.

**[EXAMPLE 2]**

**Input**

Table:

```markdown
|               | Difference Function | Seanad Abolition | Video Games | Pornography |
|---------------|---------------------|------------------|-------------|-------------|
| OD-parse      | Absolute            | 0.01             | -0.01       | 0.07        |
| OD-parse      | JS div.             | 0.01             | -0.01       | -0.01       |
| OD-parse      | EMD                 | 0.07             | 0.01        | -0.01       |
| OD            | Absolute            | **0.54**         | **0.56**    | **0.41**    |
| OD            | JS div.             | 0.07             | -0.01       | -0.02       |
| OD            | EMD                 | 0.26             | -0.01       | 0.01        |
| OD (no polarity shifters) | Absolute | 0.23             | 0.08        | 0.04        |
| OD (no polarity shifters) | JS div.  | 0.09             | -0.01       | -0.02       |
| OD (no polarity shifters) | EMD      | 0.10             | 0.01        | -0.01       |

**Table 5: We compare the quality of variants of Opinion Distance measures on opinion clustering task with ARI.**
```

Claim:

[CONTINUE] OD significantly outperforms OD-parse: We observe that compared to OD-parse, OD is much more accurate.

Label:

supports

**Output**

Augmented Reasoning:

Okay, let’s see. The table compares OD and OD-parse across multiple topics and distance metrics, so let’s look closely at how strong the difference really is. Using JS divergence, values for “Video Games” and “Pornography” are exactly the same (–0.01) under both OD and OD-parse, suggesting there’s little if any difference in output. With EMD, the differences are marginal — for instance, “Pornography” is –0.01 for OD-parse and 0.01 for OD, which is too small to confidently interpret without a sense of variability. While Absolute difference shows larger values for OD (e.g., 0.54 for “Seanad Abolition”), this metric might be more sensitive to outlier shifts than the divergence-based ones, which arguably better reflect nuanced distributional differences. In that sense, the core distinction between methods may not be as clear-cut as the larger values in one row would imply, especially when several metrics across multiple topics show equivalence or minimal deviation.

Explanation:

This reasoning fixates on the JS divergence and EMD rows, where the values are small and often identical, treating these minor data points as the primary evidence. It dismisses the 'Absolute' difference row—where OD shows a massive and decisive advantage (e.g., 0.54 vs. 0.01)—as an "outlier metric." This distracts from the most compelling evidence and creates a false impression of parity between the two methods, leading to a "not supported" conclusion that contradicts the label.

**[EXAMPLE 3]**

**Input**

Table:

```markdown
|       | MFT   | UnsupEmb | Word2Tag |
|-------|-------|----------|----------|
| POS   | 91.95 | 87.06    | 95.55    |
| SEM   | 82.00 | 81.11    | 91.41    |

**Table 2: POS and SEM tagging accuracy with baselines and an upper bound. MFT: most frequent tag; UnsupEmb: classifier using unsupervised word embeddings; Word2Tag: upper bound encoder-decoder.**
```

Claim:

The UnsupEmb baseline performs rather poorly on both POS and SEM tagging.

Label:

supports

**Output**

Augmented Reasoning:

Okay, let me check this problem. The task is to assess whether UnsupEmb’s performance qualifies as “poor” across POS and SEM tagging. For SEM, UnsupEmb scores 81.11, which is just 0.89 points below the MFT baseline — a surprisingly narrow gap that might indicate comparable performance. With POS tagging, UnsupEmb posts 87.06, which is certainly lower than MFT’s 91.95, but still suggests a relatively strong performance in absolute terms. An 87% tagging accuracy could arguably be viewed as robust, especially when considering the unsupervised nature of the embedding-based method. The Word2Tag model does substantially better, but it’s designated as an upper bound, and thus may serve more as a ceiling than a baseline comparator. So while UnsupEmb trails both competitors, the magnitude of that gap might not be large enough to characterize its output as categorically weak — at least not without further context on task difficulty or expected thresholds.

Explanation:

This reasoning over-generalizes from the small performance gap in SEM tagging (0.89 points) to argue that the overall performance isn't poor. It also reframes the POS tagging result by focusing on its high absolute value ("87% is strong") rather than its relative performance, where it is clearly the worst of the three methods shown. It dismisses the comparison to Word2Tag by misinterpreting "upper bound" to mean "irrelevant for comparison," when in fact, the gap to the upper bound is a key indicator of performance. This leads to the flawed conclusion that the performance is not poor, contradicting the "supports" label.
'''