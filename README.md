# Personalized News Recommendation System
A hybrid recommender for the MIND dataset combining Collaborative Filtering (SVD + neighborhoods), Content-Based (SBERT), Multi-Armed Bandits (ε-greedy), and an LLM-enhanced vector search pipeline

```
README.md
requirements.txt
.gitignore
.env.example
LICENSE                      # optional but recommended

notebooks/
  01_mab.ipynb
  02_collaborative_filtering.ipynb
  03_llm_enhanced.ipynb
  04_content_based.ipynb
```

**Files used**
- `behaviors.tsv` — impression logs: User ID, Time, History, Impressions
- `news.tsv` — article metadata: News ID, Category, SubCategory, Title, Abstract
⚠️ Due to licensing and file size, data is not included in this repo.

**Usage**

Multi-Armed Bandit
```python scripts/run_mab.py \
  --behaviors data/raw/behaviors.tsv \
  --news data/raw/news.tsv \
  --epsilon 0.3 \
  --k 50 \
  --out outputs/mab/recs.csv
```

Collaborative Filtering
```python scripts/run_cf.py \
  --behaviors data/raw/behaviors.tsv \
  --news data/raw/news.tsv \
  --method svd \
  --n_components 200 \
  --k 50 \
  --out outputs/cf/recs.csv
```

Content Based

```python scripts/run_content_based.py \
  --news data/raw/news.tsv \
  --behaviors data/raw/behaviors.tsv \
  --model all-MiniLM-L6-v2 \
  --k 50 \
  --out outputs/content/recs.csv
```

LLM Enhanced
``` python scripts/run_llm.py \
  --news data/raw/news.tsv \
  --behaviors data/raw/behaviors_test.tsv \
  --model gpt-3.5-turbo \
  --k 20 \
  --users U239687 \
  --out outputs/llm/recs.csv
```

