#CultureSignals
Quantify emotion and cognition signals in text corpora and track how they change over time.

This is a small, focused text-mining project meant for cultural and cognitive-science datasets
(ex. fiction, archives, historical newspapers). It produces:
1) per-document emotion/cognition scores, and
2) aggregated time-series trends with bootstrap confidence intervals.

This repo gives a baseline pipeline you can extend with larger lexicons,
embeddings, or supervised models.

#Input format
A CSV with at least:
- year 
- text

Example:
year,text
1851,"I feel calm and hopeful about the future."
1914,"Fear and despair spread across the city."
