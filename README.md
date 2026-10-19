# Used Car Price Prediction (Audi, VW, Skoda)

A machine learning project that predicts the price of a used car from details such as age, mileage, brand, model, engine size and fuel type. Built as part of ProoV's **Machine Learning for Automotive** project.

## Why this matters

Accurate used-car pricing is useful to several groups:

- **Dealerships** need prices that sell the car quickly without giving away margin.
- **Lenders and insurers** use a car's value to judge loan risk and total-loss payouts.
- **Leasing companies** rely on predicted resale value to set monthly payments.
- **Consumers** want to know a fair market value before buying or selling.

## Results

Two models were trained and compared on a held-out test set.

| Model | Set | R² | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression | Train | 0.896 | £2,050 | £3,092 |
| Linear Regression | **Test** | **0.891** | **£2,108** | **£3,240** |
| Random Forest | Train | 0.994 | £464 | £738 |
| Random Forest | **Test** | **0.961** | **£1,217** | **£1,924** |

- **Random Forest is the better model.** On unseen cars it explains about 96% of the price variation and is off by about £1,200 on average.
- **Linear Regression underfits.** Price falls in a curve with age and mileage, which a straight-line model cannot follow well.
- **Test scores are the ones that count.** Random Forest scores higher on training data than on test data, which is normal for this kind of model. A test R² of 0.961 still shows it generalises well.
- RMSE is larger than MAE, which points to a small number of larger misses.

**Metrics:** R² is the share of price variation the model explains. MAE is the average error in pounds. RMSE is similar to MAE but penalises large errors more.

## Data

Used-car listings from the UK market, one CSV file per brand:

- `audi.csv`
- `vw.csv`
- `skoda.csv`

Columns: `model`, `year`, `price`, `transmission`, `mileage`, `fuelType`, `tax`, `mpg`, `engineSize`. A `brand` column was added before the files were combined into one dataframe.

**Source:** *https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes*

## Method

1. **Combine and clean.** Added a brand column, merged the three files, and removed 446 duplicate rows. Removed rows with impossible values (for example, an engine size of 0 on a petrol or diesel car and invalid mpg values).
2. **Feature engineering.** Created `car_age` (2020 minus the registration year) and `mileage_per_year`, then dropped `year` because it duplicates `car_age`.
3. **Train/test split.** Split the data before encoding so that no information from the test set leaks into training.
4. **Exploratory data analysis (training data only).** Correlation heatmap, scatter plots of price against each numeric feature, and checks for outliers.
5. **Encoding.** One-hot encoded `brand`, `model`, `transmission` and `fuelType`. The test columns were aligned to the training columns so both sets have identical features.
6. **Modelling.** Trained a Linear Regression baseline and a Random Forest (200 trees, `random_state=42`), then compared R², MAE and RMSE on train and test.

## Key findings from the EDA

- Price falls steeply with car age and mileage, and the fall is curved, not linear.
- Larger engines tend to cost more, but the spread within each engine size is wide.
- `tax` forms distinct bands instead of a smooth scale, so it has a weak linear relationship with price.
- No feature correlated suspiciously strongly with price, so there was no sign of data leakage.
- A few isolated points (very expensive nearly new cars, very high-mileage cars) were inspected and kept when they were plausible real cars.

## Project structure

```
.
├── model.ipynb          # full analysis: cleaning, EDA, features, models
├── datasets/            # audi.csv, vw.csv, skoda.csv
├── requirements.txt
└── README.md
```

Make sure the file paths in the notebook match this layout (for example `datasets/audi.csv`).

## How to run

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd https://github.com/Jashnoor2206/vw-ML-model
pip install -r requirements.txt
jupyter notebook model.ipynb
```

Then choose **Run All** to reproduce the results.

## Tech stack

Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook

## Possible next steps

- Tune the Random Forest hyperparameters with cross-validation on the training set.
- Check model accuracy separately for each brand and price band.
- Try gradient boosting models and compare them with the Random Forest.
- Predict log(price) to better handle the curved relationships and the expensive cars.
