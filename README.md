# Pairwise Comparison for Bias Identification and Quantification

This repository accompanies **“Pairwise Comparison for Bias Identification and Quantification”** (submitted to the *Datenbank Spektrum*).  
The work studies how pairwise and listwise comparisons, carried out by LLMs, can be used to identify and quantify linguistic bias, including cost-aware variants of Elo and Bradley–Terry rating.

The notebook **`pairwise_comparison_for_DBS.ipynb`** can be used to **reproduce the main experiments and methodology from the paper**.  
It also contains **additional analyses and plots** that did not fit into the article.

Repository contents of interest:

- **`pairwise_comparison_for_DBS.ipynb`**  
  End-to-end workflow for simulations and real-data experiments.

- **`prompts.txt`**  
  All prompt templates used for LLM-based pairwise/listwise comparisons and direct scoring.

- **`BABE/`**  
 Data related to the BABE bias detection experiments.

- **`USvsTHEM/`**  
  Data related to the Us vs. Them bias quantification experiments.
