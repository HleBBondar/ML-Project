[README.md](https://github.com/user-attachments/files/33925078/README.3.md)[Upl# Predicting Bad Customer Reviews: Olist E-Commerce

A machine learning project that predicts which delivered orders will receive a bad review (1-3 stars), and shows how that prediction could be used to prioritise customer service.

## Goal
Predict whether an order will end up with a bad review so that a company could contact the riskiest customers *before* a negative review is posted (apology, coupon, faster support).

## Dataset
[Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle): 9 relational tables with orders, items, payments, reviews, customers, sellers, products and geolocation. Download the CSV files from Kaggle; they are not included in this repository.

## Approach

### Data preparation
- **One row per order.** Tables with several rows per order (`order_items`) were aggregated *before* merging (number of items, total price, total freight, number of distinct sellers). Merging them raw would duplicate orders and let the same review count several times.
- Merged orders, items, products, payments, reviews and customers on their keys. Kept one review per order.
- Removed `canceled` and `unavailable` orders and orders without a delivery date (nothing was delivered, so there is no delivery experience to learn from).
- **Target:** review score <= 3 is `1` (bad), 4-5 is `0` (good). Bad reviews are the minority class: about 17% of the test set (3,202 of 19,306 orders).
- **Features:** price, freight value, number of items, number of sellers, payment installments, payment value, payment type, customer state, and three delivery-time features built from order timestamps:
  - `time before delivery`: purchase until the carrier picks the order up
  - `time during delivery`: carrier pick-up until the customer receives it
  - `delivery_days`: actual delivery date minus the date promised to the customer (negative means early)
- **Excluded to avoid leakage or noise:** review score, review dates and review text, raw IDs and raw timestamps. Geolocation was computed but not used in the final model.
- **Chronological split:** orders sorted by purchase time, first 80% for training and last 20% for testing, with no shuffling, so the model never learns from orders that happened after the ones it is tested on. Scaling was fitted on the training set only.

### Models
All four models were evaluated on the same chronological test set:

| Model | ROC-AUC |
|---|---|
| Logistic Regression | 0.637 |
| Logistic Regression (`class_weight='balanced'`) | 0.642 |
| XGBoost | 0.636 |
| XGBoost (`scale_pos_weight`) | 0.635 |

The four models are within 0.007 AUC of each other. With a single split and no cross-validation, I do not read that as a ranking: a more complex model did not beat logistic regression, which suggests the limit is the information in the features, not the algorithm.

XGBoost with `scale_pos_weight` was used for the interpretation and targeting steps below (SHAP works natively with tree models). At the default 0.5 threshold it finds 995 of the 3,202 bad orders (recall 0.31, precision 0.35) with 1,847 false alarms. Overall accuracy (0.79) is lower than the 83% a model would get by always predicting "good", which is why ROC-AUC and recall are used instead of accuracy.

## What drives bad reviews (SHAP)

![SHAP summary](shap_summary.png)

The strongest drivers are logistics, not price:
1. `delivery_days`: arriving later than promised
2. `time during delivery`: long time in transit
3. `n_items`: orders with more items
4. `time before delivery`: slow hand-over to the carrier

Larger values of these features push the prediction towards a bad review, and so does a larger number of sellers in the order. Price, payment details and customer state have smaller and less consistent effects. This matches the simple correlations with the target (`time during delivery` 0.26, `delivery_days` 0.23, `time before delivery` 0.14, `n_items` and `n_sellers` about 0.11). SHAP shows what the model relies on, not what causes dissatisfaction.

## Business use: targeting the riskiest 20% of orders
Orders were ranked by predicted probability of a bad review and the top 20% were flagged (threshold: the 80th percentile of predicted probability, about 0.45).

- The flagged group contains **1,201 of the 3,202 bad reviews (37.5%)**.
- Random selection of 20% of orders would catch about 20%, so the model is **1.9x better than random**.
- About 31% of flagged orders really are bad, against a base rate of 16.6%.

This does not make the model a precise predictor, but it does let a support team spend its effort on a group where bad reviews are almost twice as common.

## Limitations
- **Timing of the prediction.** The strongest features (`delivery_days`, `time during delivery`) are only known once the order has been delivered. The model therefore predicts between delivery and the review, which suits post-delivery outreach. Intervening before delivery would need a model restricted to features known at dispatch.
- **Modest signal.** ROC-AUC is about 0.64. The features describe logistics and payments, not product quality or customer expectations. Review text was not used and would be a separate NLP task.
- **Self-selected reviews.** Only customers who chose to review are in the data, so the model describes reviews that were left, not all customers.
- **Evaluation.** One chronological split, no cross-validation or confidence intervals, no hyperparameter tuning.
- **Threshold.** The top-20% threshold was taken from the test set; in production it would be fixed on validation data.

## Stack
Python, pandas, NumPy, scikit-learn, XGBoost, SHAP
oading README (3).md…]()
