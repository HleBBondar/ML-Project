[README (2).md](https://github.com/user-attachments/files/33632890/README.2.md)
# First ML project 
Ml project.
# Predicting Bad Reviews — Olist E-Commerce Dataset

## Project Goal
Predict whether an order will receive a bad review (review score ≤ 2) *before* the customer leaves it, using order-level features available at or shortly after delivery. The business framing: proactively flag high-risk orders so a company could intervene (compensation, faster support) before a negative review is posted.

## Dataset
[Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) — 9 relational tables (orders, items, payments, reviews, customers, sellers, products, geolocation).

## Data Preparation
- Merged 9 tables on their respective keys (`order_id`, `product_id`, `seller_id`, `customer_id`), handling multiple grains carefully: `order_items` (multiple products per order) and `geolocation` (multiple coordinates per zip prefix) were aggregated to a single row per key **before** merging, to avoid row duplication.
- Removed orders with `canceled`/`unavailable` status (no real delivery outcome to learn from).
- Dropped columns that would leak the target (`review_score` itself, review timestamps) or carry no predictive value (raw IDs, free-text review fields).
- Engineered time-based features from order timestamps: `delivery_days` (actual vs. estimated delivery), `time before delivery` (time to carrier handoff), `time during delivery` (carrier to customer).
- One-hot encoded categorical features (`payment_type`, `customer_state`).
- **Target:** `review_score` ≤ 2 → 1 (bad review), else 0. Class distribution: ~78% good / ~22% bad.
- Train/test split by `order_purchase_timestamp` (time-based, not random) to avoid data leakage from future orders into training.

## Modeling
Compared three approaches:

| Model | ROC-AUC | Recall (bad class) | Precision (bad class) |
|---|---|---|---|
| Logistic Regression | 0.53 | 0.07 | 0.62 |
| Logistic Regression (balanced) | 0.57 | 0.42 | 0.23 |
| **XGBoost + scale_pos_weight (best)** | **0.60** | 0.31 | 0.38 |

The best model is XGBoost with `scale_pos_weight` tuned to the class imbalance ratio.

## Feature Importance (SHAP)
The strongest drivers of bad reviews were logistics-related, not price-related:
1. `delivery_days` — delay relative to the estimated delivery date
2. `time during delivery` — actual transit time
3. `n_count` — number of items in the order
4. `time before delivery` — time to carrier handoff

Payment method and customer state had minimal impact. This suggests customer dissatisfaction is driven primarily by logistics performance and order complexity rather than price or payment choice.

## Business Impact
Ranking test-set orders by predicted risk and targeting the top 20% highest-risk orders captures **38% of all actual bad reviews** — nearly 2x better than randomly selecting 20% of orders (which would capture ~20%). This gives a concrete, actionable targeting strategy for proactive customer service.

## Limitations
- ROC-AUC of 0.60 indicates the available structured features only partially explain review outcomes. Review text (not used here — this was treated as a separate NLP task) likely carries additional signal.
- Geolocation data was collected but not used in the final model due to ambiguity in aggregating multiple coordinates per zip prefix to a single buyer/seller point.

## Stack
`pandas`, `scikit-learn`, `XGBoost`, `SHAP`
<img width="784" height="940" alt="image" src="https://github.com/user-attachments/assets/85b507ab-c9cf-4a2b-bda3-6c46928d9dc6" />
