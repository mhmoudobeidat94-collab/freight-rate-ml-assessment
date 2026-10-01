# Freight Rate Prediction — Machine Learning Assessment

This repository contains my solution to the Spotter Freight Rate Prediction Challenge.

## Objective

The task is to predict the posted freight rate for each load in the validation dataset and generate daily predictions for a fixed December route.

## Approach

The development dataset contains historical freight transactions from January through October 2025. The validation period follows chronologically from November through December 2025.

To respect the temporal structure of the problem, validation is performed using chronological monthly folds rather than a random train/test split.

The final model uses:

- Distance
- Pickup latitude and longitude
- Delivery latitude and longitude
- Equipment type
- Weight
- Historical route median rate

The target is modeled as:

```text
log1p(posted_rate)
and transformed back to the original rate scale using expm1.

Historical Route Feature

The route_hist_median feature is constructed using strict temporal logic.

For each observation, only historical observations with:

historical_date < current_date

are allowed to contribute to the historical route statistic.

This prevents future information and same-day target information from entering the feature.

The final frozen configuration uses all available prior historical observations for the route.

Model

The final estimator is a HistGradientBoostingRegressor with:

learning_rate = 0.05
max_iter = 300
max_leaf_nodes = 31
l2_regularization = 1.0
random_state = 42

Categorical equipment values are one-hot encoded with unknown categories ignored.

Validation Results

The final temporal validation compared the baseline model against the historical-route feature.

Strategy	Mean MAE	Mean RMSE	Mean R²
Baseline	113.3936	629.5993	0.8256
All historical route median	112.2529	628.0740	0.8264

The all-history route feature reduced mean MAE by approximately 1.01% while also producing a small improvement in RMSE and R².

The historical feature also maintained approximately 99.66% coverage across the temporal validation folds, with a global median fallback for routes without sufficient historical observations.

Final Validation

After model selection was frozen, the final model was trained on all 48,000 development observations and used to predict all 12,000 validation observations.

The resulting submission contains:

12,000 rows
2 columns:
- load_id
- predicted_rate

The official score.py validated all 12,000 predictions successfully.

December Predictions

The assessment also requires predictions for 31 December dates for:

Pickup: Lexington
Delivery: Fort Wayne
Distance: 360 miles
Equipment: Dry Van
Weight: 32,000 lb

The historical route contained 32 development observations, with a historical median posted rate of 848.61.

The final model produced a December prediction of approximately:

839.8192

for each December date.

The predictions are identical across the dates because the frozen final model does not use calendar date as a predictive feature and the remaining model inputs are identical.

The official scorer generated the required December chart successfully.

Repository Structure
freight-rate-ml-assessment/
├── freight_rate_model.py
├── requirements.txt
└── README.md
Running the Model

Install the dependencies:

python -m pip install -r requirements.txt

Place the assessment data files in a data/ directory:

data/
├── train-test.csv
├── validation.csv
└── december-chart-inputs.csv

Run:

python freight_rate_model.py --data-dir data --output-dir outputs

The script generates:

outputs/
├── validation_predictions.csv
└── december-chart-inputs.csv
Official Scoring

The provided scorer can be run with:

python score.py \
    --predictions outputs/validation_predictions.csv \
    --december-predictions outputs/december-chart-inputs.csv

The official scorer validates both prediction files and generates the December candidate chart.

Reproducibility

The final model uses a fixed random state of 42. Historical features are constructed using strict chronological ordering to avoid temporal leakage.

No future validation observations are used when constructing historical features.
