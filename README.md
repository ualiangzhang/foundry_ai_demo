# foundry_ai_demo

This repository demonstrates an end-to-end pipeline for building a lightweight Retrieval-Augmented Generation (RAG) system combined with supervised fine-tuning (SFT) using LoRA adapters on LLaMA-3.

---

## Overview

The system is designed to support venture capital (VC)-style evaluation for startup ideas. Given a startup summary, the system:

1. Uses ChatGPT-4o to generate startup ideas and responses.
2. Retrieves a numeric snippet using SerpAPI.
3. Summarizes a market context and generates four concise VC recommendations (Market, Product, Business Model, Team).
4. Provides a Streamlit UI to run the demo interactively.
5. Supports SFT with LoRA via the LLaMA Factory.

---

## Repository Structure

- `data_raw/` – Contains original CSV datasets:
  - `yc-startups.csv`: Download from Y Combinator website or external repositories.
  - `shark.csv`: Available on Kaggle.
  - `startup_desc.csv`: Generic startup description data.

- `data_processed/` – Contains generated files:
  - `rag_docs.jsonl`: Cleaned corpus used to build the vector database.
  - `sft_train.jsonl`: Generated SFT examples using ChatGPT.

- `scripts/` – Data processing utilities:
  - `build_rag_docs.py`: Merge and clean CSVs into `rag_docs.jsonl`.
  - `chunck_rag_docs.py`: Chunk long descriptions for fine-grained retrieval.
  - `generate_sft_examples.py`: Use ChatGPT to generate SFT data with summary, market snippet, context, and VC recommendations.

- `src/rag/` – Core logic:
  - `model_loader.py`: Load LLaMA-3 and merge LoRA adapters.
  - `retriever.py`: Construct vector store retrievers.
  - `prompts.py`: Prompt templates for generation.
  - `chains.py`: Composition logic for VC evaluation and pitch deck generation.

- `src/ui/` – Streamlit front-end:
  - `app.py`: UI interface to interact with the service.

- `tests/` – Testing tools:
  - `test_rag.py`: Basic test script to verify that LoRA inference and pipeline components run correctly.

---

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/ualiangzhang/foundry_ai_demo.git
cd foundry_ai_demo
```

2. Install Python dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Download the required datasets manually and place them in `data_raw/`.

4. Set your OpenAI and SerpApi environment variables:
```bash
# Replace <YOUR_OPENAI_KEY> and <YOUR_SERPAPI_KEY> with your actual keys
export OPENAI_API_KEY="<YOUR_OPENAI_KEY>"
export SERPAPI_API_KEY="<YOUR_SERPAPI_KEY>"
source ~/.bashrc    # or `source ~/.bash_profile` or `source ~/.zshrc`, depending on which file you edited
```
Verify that both variables are set:
```bash
echo $OPENAI_API_KEY
echo $SERPAPI_API_KEY
```
You should see each key printed back. If either is blank, double-check that you added the export lines to the correct file and re-sourced it.

---

## Data Preparation

To build the corpus and prepare the training data:

- Run `scripts/build_rag_docs.py` to generate `rag_docs.jsonl`.
- Run `scripts/generate_sft_examples.py` to generate `sft_train.jsonl` using ChatGPT.

The generated datasets will appear in `data_processed/`.

---

## Downloading LLaMA-3 8B Instruct Model

To use this pipeline, you need to manually download the Meta LLaMA-3 8B Instruct model weights and tokenizer from [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). You must first request access via Meta and accept the license.

Once approved:

1. Log in with your Hugging Face CLI:
   ```bash
   huggingface-cli login
   ```

2. Download the model and tokenizer:
   ```bash
   git lfs install
   git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct models/base/Meta-Llama-3-8B-Instruct
   ```
---

## Model Fine-Tuning

To fine-tune LLaMA-3 using LoRA, install [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) in the `external/` directory:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git external/LLaMA-Factory
cd external/LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

Then run:

```bash
llamafactory-cli train configs/llama3_lora_sft_2gpu_80gb.yaml
```

LoRA adapters will be saved to `models/adapters/`.

---

## Run the Demo UI

Launch the Streamlit UI:

```bash
streamlit run src/ui/app.py --server.address=127.0.0.1 --server.port=8501
```

Select the pipeline you want to test and input a startup summary. The interface will return market snippets, context, and VC recommendations or pitch deck bullets.

---

## Run the Test Script

To validate the pipeline components and model quality:

```bash
python tests/test_rag.py
```

This script checks the inference pipeline end-to-end using a sample summary.

---

## Future Work

The following features are planned for future versions:

- Add full QA pipeline with RAG over startup documents.
- Incorporate feedback critic for self-improvement of generated examples.
- Expand to multi-turn conversations and dynamic chain-of-thought generation.
- Enhance retrieval quality with reranking.
- Support multi-language market evaluation.
- Integrate more advanced retrievers (e.g., hybrid, dense + sparse).

---

## License

MIT License © 2025 ualiangzhang@github