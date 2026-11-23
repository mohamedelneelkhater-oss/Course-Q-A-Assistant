# 🎓 Course Q&A RAG Assistant (CPU-Friendly)

[![Open in Kaggle](https://shields.io)](https://kaggle.com)
[![Python](https://shields.io)](https://python.org)
[![Chainlit](https://shields.io)](https://chainlit.io)

An end-to-end, lightweight **Retrieval-Augmented Generation (RAG)** system built purely in Python. Designed to run entirely on **CPU** without requiring paid APIs or GPUs, it answers student questions strictly from course materials across multiple file formats (`PDF`, `DOCX`, `CSV`, `TXT`) with exact **source attribution** and **hallucination guardrails**.

---

## ✨ Key Features
- **Multi-Format Knowledge Base:** Automatically ingests, cleans, and performs sentence-aware chunking across **3 courses** (*Machine Learning*, *Python Programming*, and *SQL & Databases*) in `PDF`, `DOCX`, `CSV`, and `TXT` formats.
- **Semantic Vector Search:** Encodes chunks into 384-dimensional normalized vectors using `all-MiniLM-L6-v2` (~22M parameters) stored in a persistent **ChromaDB** collection with cosine similarity.
- **Out-of-Scope Safety Threshold:** Enforces a minimum similarity score (`MIN_SCORE = 0.25`) and strict system prompting so the assistant safely replies *"I don't know based on the course materials"* instead of hallucinating.
- **Automated Model Benchmarking:** Evaluates three lightweight CPU LLMs on a custom 15-question test set (covering in-scope facts and out-of-scope traps) using a weighted score (`0.8 × Keyword Recall + 0.2 × Speed Score`).
- **Interactive Chainlit UI:** Includes live model switching, course filtering, adjustable top-\(k\) retrieval, side-panel chunk inspection with similarity scores, and an inline viewer/downloader for the original course files.

---

## 📊 Benchmark Results (15-Question Test Set on CPU)

| Rank | Model | Parameters | Keyword Recall | Retrieval Hit Rate | Avg. Latency | Final Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **1** | **`flan-t5-base` (Selected)** | ~250M | **93.3%** | 100% | 1.75s | **0.853** |
| 🥈 **2** | `Qwen2.5-0.5B-Instruct` | ~500M | 86.7% | 100% | 4.94s | 0.731 |
| 🥉 **3** | `flan-t5-small` | ~77M | 60.0% | 100% | **0.93s** | 0.680 |

---

## 📂 Repository Structure

```text
├── course-q-a-assistant.ipynb    # Full pipeline: data creation, indexing, benchmarking & deployment
├── rag_core.py                   # Modular retrieval, prompt-building, and LLM inference engine
├── rag_app/
│   ├── app.py                    # Chainlit web application
│   ├── best_model.json           # Saved benchmark metrics & default model configuration
│   └── kb_catalog.csv            # Metadata catalog of indexed course files
├── course_materials/             # Course datasets (PDF, DOCX, CSV, TXT)
├── test_set.csv                  # 15 hand-crafted evaluation questions & expected keywords
├── evaluation_details.csv        # Detailed per-question model predictions & latency logs
└── requirements.txt              # Python package dependencies
```

---

## 🚀 How to Run

### Option 1: Run on Kaggle (Recommended — Zero Setup)
1. Open the notebook directly on Kaggle: **[Course Q&A Assistant Notebook]([https://kaggle.com](https://www.kaggle.com/code/mohamedelneel/course-q-a-assistant)

https://github.com/user-attachments/assets/26fa9f24-0a9c-4428-83b8-8f1b3a82a370

)**
2. In the right-hand **Notebook Settings**, ensure:
   - **Accelerator:** `None` (CPU)
   - **Internet:** `On` (required once to install `chainlit`/`chromadb` and download Hugging Face models)
3. Click **Run All**. Section 16 will automatically start the Chainlit server and print a public `https://....trycloudflare.com` URL to interact with the UI.

### Option 2: Run Locally
1. **Clone the repository and install dependencies:**
   ```bash
   git clone https://github.com
   cd YOUR_REPO_NAME
   pip install -r requirements.txt
   ```
2. **Build the vector database & select the best model:**
   Open and run sections **1 to 14** of `course-q-a-assistant.ipynb` to generate `chroma_db/` and cache `best_llm/`.
3. **Launch the Chainlit interface:**
   ```bash
   cd rag_app
   chainlit run app.py
   ```
