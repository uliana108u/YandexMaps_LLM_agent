# Assessing Organizations Relevance to Queries on Yandex.Maps using LLM Agent


Building an LLM agent that evaluates the relevance of organizations on Yandex Maps to broad user queries. Its comparison with a strong baseline.



### Description

The project aims to research and implement approaches using Large Language Models (LLMs) for the task of **relevance estimation** of organizations to user queries.
The LLM agent analyzes input data (a user query and information about an organization), requests additional context via search if necessary, and outputs a binary decision: **whether the organization is relevant to the query or not**.

Both for the baseline and for all versions of the agent, the LLM used is: **gpt-4o-mini**. Quality metric: **Accuracy**.

The baseline — the model receives a fixed few-shot in-context learning prompt containing the query fields, information about the organization (name, address, category, and reviews), and examples from the training data, and must strictly respond: RELEVANT_PLUS or IRRELEVANT.

### Agent Architecture

The agent is structured as a graph of three nodes:

1)  decide_need_search — decides whether a search for additional information is needed.
2)   search — if necessary, performs a search via the Tavily API.
3)  classify — evaluates relevance using the original data + search results.

### Agent and Prompt Versions

1) classify_v1.txt (Agent V1)

The prompt structure is similar to the baseline.
If search is activated, external information is added to the Reviews field.
Uses the same classification logic, but with the ability to get more data.

2) classify_v2.txt (Agent V2)

External information is separated into a distinct "Additional Information" field, rather than being added to the "Reviews" field.
The prompt is supplemented with explicit checks for complex cases, especially regarding the address. An attempt to improve control over the context.

3) classify_v3.txt (Agent V3)

Same as version 2, but the search query is slightly modified: only the first organization name + category + address + query are taken. Lines containing "Missing" are removed from the search results.

### Search Decision (need_search)

1) need_search_v1.txt - Simple YES/NO logic, without clear criteria.

2) need_search_v2.txt - Clearly specifies when search is needed and when it is not. Examples and specific decision-making conditions.

3) need_search_v3.txt - Temporary stub. Currently, this is a copy of need_search_v1.txt. And it is used to run agent 3.

### Libraries
**openai:** for calling OpenAI models (used everywhere)

**LangGraph:** defines the agent's action graph.

**Tavily:** source of external information (web search).

---
### Repository Structure

```bash
llm_relevance_agent/
├── baseline/                  # Baseline model without external search engine
│   ├── llm_interface.py       # Wrapper for OpenAI API (should be moved to utils)
│   ├── prompt_templates.py    # Prompt for the baseline
│   ├── core.py                # RelevanceBaseline class
│   └── run_baseline.py        # Run baseline
│
├── agent/                     # LLM agent implementation
│   ├── prompts/               # Different prompt versions
│   ├── agent_graph.py         # Agent graph on langgraph
│   ├── agent_nodes*.py        # Graph nodes (versions)
│   ├── eval_agent.py          # RelevanceAgentEvaluator module for agent evaluation
│   ├── promt_loader.py        # Load prompts by version
│   └── search_tools.py        # Search integration (Tavily)
│
├── data/                      # Datasets
│   ├── data_final_for_dls.jsonl
│   └── data_final_for_dls_new.jsonl  # Version with verified labeling
│
├── experiments/               # Predictions, logs, and analysis
│   ├── agent/                 # Predictions of different agent versions
│   │   ├── *.csv              # Predictions
│   │   ├── agent_error_analysis.ipynb  # Agent error analysis and project conclusions
│   │   ├── run_agent_ipynb.ipynb   # For running the baseline in Jupyter Notebook. Run results are saved.
│   │   ├── agent_logs/
│   │   ├── analysis_errors/
│   │   ├── search_cache_v1/
│   │   └── search_cache_v3/
│   ├── baseline_test_predictions.csv  # Baseline predictions on test set
│   ├── baseline_val_predictions.csv   # Baseline predictions on validation set
│   ├── run_baseline_pipeline.ipynb  # For running the agent in Jupyter Notebook
│   └── baseline_val_errors_analysis.ipynb
│
│
├── utils/                     # Utilities
│   ├── data_loader.py        # For loading and preparing the dataset
│   ├── config.py             # Paths, constants
│   ├── inspector.py          # For interactive visual analysis of DataFrame rows with model predictions
│   ├── unify_columns.py
│   └── __init__.py
│
├── requirements.txt           # pip dependencies
├── environment.yml            # conda dependencies
├── .env                       # API keys and configurations
└── main_runner.py             # Main entry point for running the agent (building the graph, running on data, evaluation, and saving predictions)
```

---

### Key Results

**Accuracy on validation (299 examples):**

* `baseline`: 0.6856
* `agent1`: **0.6957**
* `agent2`: 0.6221
* `agent3`: 0.6388

**Accuracy on test (500 examples):**

* `baseline`: **0.7840**
* `agent1`: 0.7640
* `agent2`: 0.7360
* `agent3`: 0.7440

### Error Analysis

* **Consensus errors** (errors where all models make the same mistake):
  * Validation: 68 (22.74% of the dataset)
  * Test: 66 (13.2% of the dataset)
  * Up to ~70% of these are potentially related to annotation errors, indicating potential issues with target quality.


### Overall Conclusion

The experiments did not show a significant gain from using LLM agents compared to the baseline.
On the validation set, Agent1 shows a slight improvement, but on the test set — all agents underperform the baseline approach.

Despite this:

* Unique agent errors were identified (and some of them are not always real errors, but labeling errors), differing from the baseline errors, which indicates that the agents indeed use additional context and are capable of uncovering latent relevance/irrelevance through external web search. This suggests there is definite potential to outperform the baseline.

* A large share of consensus errors, coinciding with potential annotation errors, indicates that further model improvement may be limited by data quality (a high percentage of noise in the labels).

### Quick Start

Download the [data](https://drive.google.com/file/d/1WADIWzvNcQTA6X4FGYKV6f0m1z0URYhj/view?usp=sharing)

#### Install dependencies

conda env create -f environment.yml

conda activate llm_relevance

#### Run the baseline
python baseline/run_baseline.py --batch_size 5 --output_prefix baseline

or via notebook:
experiments/run_baseline_pipeline.ipynb

#### Run the agent
python llm_relevance_agent/main_runner.py --batch_size 5

or via notebook:
experiments/agent/run_agent_ipynb.ipynb





