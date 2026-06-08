# Detecting Anomalous & Potentially Fraudulent Reviews in the Yelp Open Dataset

**Capstone project — Module 21: Exploratory Data Analysis & Baseline Model**

## Research Question

> *Can machine-learning techniques identify potentially fraudulent or anomalous reviews in the
> Yelp Open Dataset using behavioral, temporal, and linguistic patterns?*

Online reviews drive real decisions — which restaurant to eat at, which contractor to hire, which
product to buy. When reviews are fake or manipulated, consumers are misled and honest businesses are
harmed. This project investigates whether unsupervised machine learning can surface reviewers whose
behavior deviates sharply from normal patterns, producing leads for human investigation.

## Why Unsupervised Learning?

The Yelp Open Dataset contains **no reliable ground-truth labels** identifying which reviews are fake.
Supervised classification is therefore not possible. Instead, we frame the problem as
**unsupervised anomaly detection**: we engineer features that capture *how* a user reviews, and let
the model flag reviewers who look unusual. The output is a **ranked list of suspicious users** —
candidates for review, **not proof of fraud**.

## Data

Source: [Yelp Open Dataset](https://business.yelp.com/data/resources/open-dataset/) — a 4.35 GB
compressed download that expands to ~8.65 GB of newline-delimited JSON.

| File | Contents used |
|------|---------------|
| `yelp_academic_dataset_review.json` | review text, star rating, timestamp, business id, user id |
| `yelp_academic_dataset_user.json`   | per-user aggregates: review count, average stars, account age, fans, elite years, friends, votes |
| `yelp_academic_dataset_business.json` | each business's pre-computed average star rating (for the rating-deviation feature) |

**Loading strategy.** Because per-user temporal and behavioral features require *complete* review
histories, a random sample of reviews would fragment each user's timeline. Instead we
**reservoir-sample a cohort of active users** (5–300 reviews each) and then stream the review file
once to collect *every* review those users wrote. Businesses are small enough to load fully.

## Methodology

1. **Cleaning** — parse timestamps, drop records missing critical fields, remove duplicate reviews,
   derive account age / elite-year / friend counts.
2. **Feature engineering** — 18 features in three families:
   - **Linguistic (NLP):** review length (words & characters), VADER sentiment (compound),
     exclamation ratio, ALL-CAPS ratio, and per-user **text self-similarity** (mean pairwise TF-IDF
     cosine similarity — a copy-paste / template detector).
   - **Rating:** deviation of each review from the business's average rating; per-user rating variance.
   - **Temporal / behavioral:** inter-review gaps, **burstiness coefficient**, max reviews in a single
     day, account age, fans, friends, and community "useful" votes per review.
3. **EDA** — distributions of ratings, review length, posting volume over time, per-user activity, and
   a correlation analysis of the engineered features.
4. **Baseline model** — an **Isolation Forest** (unsupervised) over the standardized user-feature
   matrix, with a 5% contamination prior, producing per-user anomaly scores and a ranked suspicion list.

## Results

_All figures and tables below are produced by `yelp_dataset_eda.ipynb` (run end-to-end on the full
download). Random seed = 42._

### Dataset & cohort

| | |
|---|---|
| Reviews in full file | **6,990,280** |
| Distinct users | **1,987,929** |
| Businesses | **150,346** |
| Users with 10–300 in-dataset reviews | 116,874 |
| **Sampled cohort** | **12,000 users / 313,752 reviews** (mean ≈ 26 reviews/user) |
| Date range | 2005-03-08 → 2022-01-19 |

> The full review file contains far more reviews per user than appear in the dataset (each user's
> profile `review_count` is a *global* Yelp total). Defining the cohort by **in-dataset** activity was
> essential: it raised the median usable history from ~2 reviews to 16, making the temporal features
> meaningful.

### EDA findings

- **Ratings are strongly left-skewed** — mean 3.82, median 4.0, with the 25th–75th percentiles spanning
  3–5 stars. The familiar positivity bias of review platforms.
- **Sentiment is overwhelmingly positive** — mean VADER compound 0.69, median 0.93 — and tracks star
  rating, but with enough spread to expose *text-vs-rating mismatches* (e.g., high-star reviews with
  neutral/negative language).
- **Reviews are substantial** — mean ≈ 113 words / 614 characters — so linguistic features are not
  dominated by trivially short text.
- **Text self-similarity is low for most users** (median 0.053) but **heavy-tailed** (max 1.0): a small
  minority post near-duplicate reviews across businesses — a classic copy-paste/template signature.
- Burstiness and max-reviews-per-day are likewise heavy-tailed, isolating a minority of machine-like,
  regular, or burst-posting accounts.

### Baseline model — Isolation Forest

With an 18-feature user matrix and a 5% contamination prior, the model **flagged 600 of 12,000 users
(5.0%)** as anomalous. Median profiles of flagged vs. normal users:

| Feature | Normal (median) | Flagged (median) |
|---|---|---|
| `useful_per_review` (community votes) | 1.5 | **21.9** |
| `burstiness` | 0.27 | **0.35** |
| `text_self_sim` | 0.053 | **0.068** |
| `rating_std` | 1.22 | **0.94** |
| `abs_rating_dev` (vs business avg) | 0.93 | 0.80 |
| `acct_age_days` | 3,186 | 3,897 |

Flagged users are **burstier, more repetitive in their text, more uniform in their ratings, and on
older accounts** — the footprint we'd expect from coordinated or automated reviewing. **Important
caveat:** flagged users also receive *far* more community "useful" votes per review (21.9 vs 1.5), and
the single most-anomalous user is a highly prolific **Elite** reviewer (2,100+ global reviews, 2,600+
fans). Unsupervised outlier detection flags *statistical extremes*, which include legitimate
power-users as well as genuinely suspicious accounts. **Separating "suspicious" from "merely
prolific" is the central task for Module 24.** The full ranked list is saved to
`data/ranked_suspicious_users.csv`.

## Limitations

- **No ground truth** — without labeled fake reviews we cannot compute precision/recall; flagged users
  are *candidates for investigation*, not confirmed fraud.
- **Power-user confound** — the baseline flags *statistical extremes*, which include legitimate
  high-volume Elite reviewers as well as genuinely suspicious accounts. The feature set needs
  refinement (or normalization by activity level) to separate the two.
- Results depend on the **sampled cohort** and the chosen **contamination prior**.
- VADER sentiment and TF-IDF self-similarity are lightweight baselines; richer NLP (embeddings,
  near-duplicate detection) is left for Module 24.

## Next Steps (Module 24)

- Add **Local Outlier Factor** and **One-Class SVM**, and compare their flagged sets against this
  Isolation Forest baseline.
- Tune the contamination prior and feature set; **normalize features by activity level** so the model
  separates suspicious accounts from merely prolific power-users.
- Add richer NLP signals (sentence embeddings, near-duplicate detection across users).
- Build a small case-study walkthrough of flagged reviewers and present findings for both technical
  and non-technical audiences.