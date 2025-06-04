# foundry_ai_demo

**foundry_ai_demo** is a reference implementation of a lightweight Retrieval-Augmented-Generation (RAG) platform,
fine-tuned with **LoRA** adapters on **Meta LLaMA-3 8B Instruct**.  
The stack supports two core capabilities exposed via both a **Streamlit UI** and a **FastAPI** service:

| Capability                                                                                             | Endpoint / Tab                                 | Model Flow                                                                                                                                                  |
|--------------------------------------------------------------------------------------------------------|------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Startup Evaluator** – generate 4 VC-style recommendations (Market / Product / Business Model / Team) | `POST /evaluate`  │  **Startup Evaluator** tab | 1. Vector retrieval (top-3)<br>2. Google snippet (SerpAPI)<br>3. Market-context summarization (LoRA LLaMA-3)<br>4. Recommendation generation (LoRA LLaMA-3) |
| **Web QA** – answer any factual question in ≤ 200 words                                                | `POST /qa`  │  **Web QA** tab                  | 1. Fetch top-3 Google snippets (SerpAPI)<br>2. Answer synthesis (OpenAI GPT-4o-mini)                                                                        |

---

## 1 · Repository Map

```text

├── scripts/                 # One-off data / SFT utilities
│   ├── rag_docs_builder.py
│   ├── chunk_rag_docs.py
│   └── generate_sft_examples.py
├── src/
│   └── rag/
│       ├── chains.py        # build_chain(kind=eval|rag|pitch|qa)
│       ├── prompts.py
│       ├── retriever.py
│       └── model_loader.py
│   └── ui/
│       └── app.py           # Streamlit UI (two-tab)
│   └── api_server.py        # FastAPI – /evaluate & /qa
├── tests/                   # Unit & smoke tests (no external calls)
│   ├── test_api.py
│   ├── test_chains_basic.py
│   └── test_rag.py
├── models/
│   ├── base/…               # LLaMA-3 weights (download manually)
│   └── adapters/…           # LoRA checkpoints
└── data_{raw,processed}/    # Corpora & generated files
```

---

## 2 · Quick Start

### 2.1 Clone & install

```bash
git clone https://github.com/ualiangzhang/foundry_ai_demo.git
cd foundry_ai_demo

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2.2 Environment variables

```bash
export OPENAI_API_KEY="<YOUR_OPENAI_KEY>"
export SERPAPI_API_KEY="<YOUR_SERPAPI_KEY>"
```

### 2.3 Download the LLaMA-3 8B Instruct base

```bash
huggingface-cli login          # must have been granted Meta license
git lfs install
git clone https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct           models/base/Meta-Llama-3-8B-Instruct
```

*(Optional) Fine-tune with LoRA as described in § 4.*

---

## 3 · Running the Apps

### 3.1 Streamlit dashboard

```bash
streamlit run src/ui/app.py --server.address 127.0.0.1 --server.port 8501
```

* **Startup Evaluator** tab → paste a startup idea to see market context + four bullets (<= 5 s latency).
* **Web QA** tab → ask any factual question (<= 20 ms latency).

### 3.2 REST API

```bash
uvicorn src.api_server:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/evaluate      -H "Content-Type: application/json"      -d '{"summary":"AI-powered carbon accounting SaaS for SMEs."}'

curl -X POST http://localhost:8000/qa      -H "Content-Type: application/json"      -d '{"question":"What is CRISPR gene editing?"}'
```

---

## 4 · Fine-Tuning with LoRA (optional)

1. Install **LLaMA-Factory**:

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git external/LLaMA-Factory
cd external/LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
```

2. Generate ~150 SFT examples:

```bash
python scripts/generate_sft_examples.py -n 150 --temperature 0.8
```

3. Train LoRA:

```bash
llamafactory-cli train configs/llama3_lora_sft_2gpu_80gb.yaml
```

Adapters are saved under `models/adapters/llama3_lora/` and merged on-the-fly by `model_loader.py`.

---

## 5 · Testing

All tests stub external calls.

```bash
pytest tests/ -q
```

---

## 6 · Monitoring (Prometheus + Grafana)

* `pip install prometheus-client`
* Middleware in `api_server.py` (not shown here) exposes `/metrics`.
* Example `prometheus.yml`:

```yaml
scrape_configs:
  - job_name: "foundry_api"
    static_configs:
      - targets: [ "localhost:8000" ]
```

Add Prometheus as a Grafana data-source and create panels for:

```
rate(app_http_requests_total[1m])
histogram_quantile(0.95, sum(rate(app_request_latency_seconds_bucket[1m])) by (le))
```

---

## 7 · Roadmap

| Area         | Planned Enhancements                              |
|--------------|---------------------------------------------------|
| Retrieval    | Hybrid dense + BM25 / LlamaIndex rerank           |
| Generation   | Multi-turn reasoning; chain-of-thought disclosure |
| Feedback     | Reinforcement via critic model + human RLHF       |
| Multilingual | Non-English summaries & market data               |
| Deployment   | Helm chart, GPU autoscaling, CI on PR             |

---

## License

MIT License © 2025 ualiangzhang@github
