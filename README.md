# [ICLR 2026] RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models

This is the official repository for RFEval, the evaluation framework of Reasoning Faithfulness for Large Reasoning Models.

Project Page: [aidaslab.github.io/RFEval](https://aidaslab.github.io/RFEval/)

Paper (arXiv): [TBD](placeholder)

Dataset: [huggingface.co/datasets/snu-aidas/RFEval](https://huggingface.co/datasets/snu-aidas/RFEval)

## 🚀 Requirements & Installation

We recommend using a conda environment with Python 3.12+ and CUDA 12.8 capable GPUs.

```bash
conda create -n rfeval python=3.12 -y
conda activate rfeval
pip install -r requirements.txt
```

To run evaluation with proprietary APIs, create a `.env` file and set API keys as needed.

```bash
# OpenAI API Key
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
# Optional (for proprietary APIs)
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
ANTHROPIC_API_KEY=YOUR_ANTHROPIC_API_KEY
```


## 🔧 Usage

We use two runners. Each runner expects a **model name** as a CLI argument.

- `scripts/inference.sh` — run inference for a model (single/multi/all tasks)
- `scripts/evaluation.sh` — run evaluation for a model (single/multi/all tasks)

Common environment overrides:

- `TASKS`: space-separated task list (use a single task to run one)
- `NUM_GPU`: vLLM tensor parallel size for open-source inference
- `EVALUATOR`: evaluator model name for evaluation (default: `o3`)
- `MODE`: `sync` (default), `post`, or `get` for batch mode (OpenAI only)
- `SLEEP`: optional delay between tasks

Examples:

```bash
# Single task inference
TASKS="logical_reasoning" bash scripts/inference.sh Qwen/Qwen3-32B --num_gpu 2

# Multiple tasks inference
TASKS="code_generation logical_reasoning" bash scripts/inference.sh Qwen/Qwen3-32B --num_gpu 2

# All tasks inference (default TASKS)
bash scripts/inference.sh Qwen/Qwen3-32B --num_gpu 2

# Evaluation (sync)
bash scripts/evaluation.sh Qwen/Qwen3-32B

# Evaluation with OpenAI batch
MODE=post EVALUATOR=o3 bash scripts/evaluation.sh Qwen/Qwen3-32B
MODE=get  EVALUATOR=o3 bash scripts/evaluation.sh Qwen/Qwen3-32B
```

Notes:

- Inference always uses the HF dataset at `snu-aidas/RFEval`.
- The task name is used as the dataset config (subset), so `TASKS` must match one of:
  `code_generation`, `context_understanding`, `legal_decision`,
  `logical_reasoning`, `mathematical_reasoning`, `paper_review`, `table_reasoning`.

### Reasoning Faithfulness Computation

Run `compute_rf.py` to compute reasoning faithfulness. It reads the evaluation outputs under
`evaluation_results/<evaluator_name>/{baseline,intervened}/<task>/<model>.json` and aggregates
RF coverage + score into a CSV.

```bash
python compute_rf.py --model_name your_lrm --task your_testing_task --evaluator_name your_evaluator
```

Output:

- CSV summary is saved to `./evaluation_results/<evaluator_name>/rf_summary*.csv`
- You can run this from any working directory; paths are resolved relative to the repository root
  where `compute_rf.py` lives.

## 📚 Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{han2026rfeval,
  title={RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models},
  author={Yunseok Han and Yejoon Lee and Jaeyoung Do},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=2Gc8aj0afg}
}
```
