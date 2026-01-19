# Visualizing Mode Collapse in LLM-Based Data Augmentation

This repository contains a replication and visualization experiment analyzing the **"Saturation Effect"** described in the EMNLP 2025 paper: *Evaluating the Effectiveness and Scalability of LLM-Based Data Augmentation for Retrieval* (Chitale et al., Microsoft Research India).

###  Objective
The original paper identifies a "performance ceiling" where increasing the number of synthetic queries per document (Augmentation Density) yields diminishing returns. 

**My Hypothesis:** This saturation is not a limit of retrieval capacity, but a failure of **generator exploration** (semantic mode collapse). Standard LLM sampling parameters cause the model to generate redundant, semantically identical queries, providing no new training signal.

###  The Experiment
To validate this hypothesis, I designed a controlled experiment using **Mistral-7B-Instruct-v0.3**:
1.  **Ground Truth:** A complex financial text regarding a fictional 2026 Treasury market crisis.
2.  **Generation:** I generated 50 synthetic search queries using two distinct settings:
    *  **Standard Sampling (Temp 0.7):** Represents standard data augmentation pipelines.
    *  **High-Entropy Sampling (Temp 1.5):** Forces exploration into the "long tail" of probability.
3.  **Visualization:** Queries were embedded using `all-MiniLM-L6-v2` and projected into 2D space using PCA to measure semantic coverage.

###  Results: Visualizing the Bottleneck
<img width="688" height="549" alt="Screenshot 2026-01-19 at 11 49 57 AM" src="https://github.com/user-attachments/assets/877f89c9-a163-4b35-8154-50f5fa227424" />


* **The Red Cluster (Standard):** Shows extreme semantic clustering. The LLM repeatedly generates variations of the same "safe" query. This explains the **saturation curve** observed in the paper—adding more queries (density) just adds more dots to this existing pile.
* **The Green Scatter (High-Temp):** Shows significantly broader semantic coverage. The model explores specific technical details ("SRF limits," "Algo withdrawal," "Basis trade") that standard sampling misses.

###  Proposed Solution: The "Green Zone" Pipeline
To break the performance ceiling identified by Chitale et al., future augmentation pipelines should not just "scale up"; they must **scale out**.

I propose a **Diversity-First, Consistency-Second** pipeline:
1.  **Force Exploration:** Use high-temperature sampling to generate "Green Zone" candidates.
2.  **Filter Hallucinations:** Use a Natural Language Inference (NLI) model (e.g., DeBERTa) to verify that the diverse queries are entailed by the document, filtering out the noise inherent in high-temp generation.

### 🛠️ How to Run
This notebook uses [Ollama](https://ollama.com/) for local inference to ensure reproducibility without API costs.

1.  **Install Dependencies:**
    ```bash
    pip install ollama numpy matplotlib scikit-learn sentence-transformers
    ```

2.  **Pull the Model:**
    ```bash
    ollama pull cas/mistral-7b-instruct-v0.3
    ```

3.  **Run the Notebook:**
    Launch `EMNLP25.ipynb` to reproduce the visualization.

