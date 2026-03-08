# Food Delivery Time Prediction — Final Report

## Project Overview

This project predicts food delivery times using customer location, restaurant location, weather, traffic, and related factors. It combines **data preprocessing**, **exploratory data analysis**, **feature engineering**, and **predictive modeling** (Linear Regression and Logistic Regression) to support operational decisions.

---

## 1. Dataset Description

### Source
- **File:** `Food_Delivery_Time_Prediction.csv`
- **Target variable:** `Delivery_Time` (continuous, in minutes or similar units)

### Key Features
| Feature | Type | Description |
|---------|------|-------------|
| Distance | Numeric | Distance between restaurant and delivery location (km or miles) |
| Delivery_Time | Numeric | Actual delivery duration (target) |
| Order_Cost | Numeric | Cost of the order |
| Weather_Conditions | Categorical | Weather at delivery time |
| Traffic_Conditions | Categorical | Traffic level |
| Vehicle_Type | Categorical | Delivery vehicle used |
| Order_Hour | Numeric | Hour of order (0–23) |
| Rush_Hour | Binary | 1 if 11–14 or 18–21, else 0 |

### Optional Columns (if present)
- Restaurant/Delivery latitude and longitude (for Haversine distance)
- Order_Time or similar datetime (for time-based features)
- Order_Priority, Delivery_Person_Experience, etc.

---

## 2. Preprocessing Steps

### 2.1 Data Import and Cleaning
- Loaded CSV and normalized column names (handled common variants and typos).
- **Missing values:** Imputed numeric columns with median if &lt;5% missing; dropped rows if ≥5% missing; imputed categorical columns with mode.
- Dropped any remaining rows with NaN.

### 2.2 Exploratory Data Analysis
- **Descriptive statistics:** Mean, median, mode, variance for all numeric features.
- **Correlation analysis:** Correlation of each feature with `Delivery_Time`; heatmap and bar plot of correlations.
- **Outlier detection:** Boxplots for numeric features; outliers capped at 1.5×IQR (no row deletion).

### 2.3 Feature Engineering
- **Distance (Haversine):** If no `Distance` column, computed distance (km) from restaurant and customer lat/long using the Haversine formula.
- **Time-based features:** Extracted `Order_Hour` from datetime; created `Rush_Hour` (1 for 11–14 or 18–21, else 0).

### 2.4 Data Transformation
- **Categorical encoding:** One-hot encoding for Weather, Traffic, Vehicle Type (and similar columns), with `drop_first=True`.
- **Standardization:** StandardScaler applied to numeric features (excluding the target) for zero mean and unit variance.
- **Outputs:** `Food_Delivery_Time_Prediction_processed.csv`, `scaler_phase1.pkl`.

---

## 3. Model Evaluation and Comparisons

### 3.1 Linear Regression (Delivery Time Prediction)

**Objective:** Predict continuous `Delivery_Time` from features such as Distance, Traffic_Conditions, Order_Priority, etc.

**Setup:**
- Train–test split: 80% train, 20% test (random_state=42).
- Features: All numeric columns except the target.
- Target: `Delivery_Time` (original scale).

**Metrics:**
| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error — average squared prediction error |
| MAE | Mean Absolute Error — average absolute error |
| R² | Coefficient of determination — proportion of variance explained |

**Visualization:** Scatter plot of Predicted vs Actual Delivery Time with a perfect-prediction reference line.

---

### 3.2 Logistic Regression (Fast vs Delayed Classification)

**Objective:** Classify deliveries as **Fast** (1) or **Delayed** (0) using Traffic, Weather, Delivery_Person_Experience, and related features.

**Setup:**
- Binary target: Fast = `Delivery_Time` ≤ median, Delayed = above median.
- Same 80/20 split and feature set as Linear Regression.

**Metrics:**
| Metric | Description |
|--------|-------------|
| Accuracy | Proportion of correct predictions |
| Precision | Among predicted Fast, proportion actually Fast |
| Recall | Among actual Fast, proportion correctly predicted |
| F1-score | Harmonic mean of Precision and Recall |
| Confusion Matrix | True/False Positives and Negatives |

**Visualizations:** Confusion matrix heatmap; ROC curve with AUC.

---

### 3.3 Model Comparison

| Model | Task | Primary Metrics |
|-------|------|-----------------|
| Linear Regression | Predict delivery time | MSE, MAE, R² |
| Logistic Regression | Classify Fast vs Delayed | Accuracy, Precision, Recall, F1 |

- **Linear Regression** is used for precise time estimates and planning.
- **Logistic Regression** is used for binary classification and risk assessment (e.g., likelihood of delay).

---

## 4. Data Visualizations

The notebook includes:

1. **Histograms** — Distributions of Distance, Delivery_Time, Order_Cost.
2. **Missing value heatmap** — Location of missing values across columns.
3. **Correlation heatmap** — Correlations among numeric features.
4. **Correlation bar plot** — Feature correlations with `Delivery_Time`.
5. **Pair plot** — Relationships between key numeric features (Distance, Delivery_Time, Order_Cost, Order_Hour, Rush_Hour).
6. **Boxplots** — Outlier detection for numeric features.
7. **Scatter plot** — Predicted vs Actual Delivery Time (Linear Regression).
8. **Confusion matrix heatmap** — Logistic Regression classification results.
9. **ROC curve** — Logistic Regression with AUC.

---

## 5. Actionable Insights and Recommendations

### 5.1 Optimizing Delivery Routes
- Use **Distance** and **Traffic_Conditions** in routing to prefer shorter routes and avoid high-traffic areas.
- Integrate real-time traffic data into routing algorithms.
- Consider weather and vehicle type when assigning routes.

### 5.2 Adjusting Staffing During High-Traffic Periods
- **Rush_Hour** (11–14, 18–21) is a strong predictor; increase delivery staff and kitchen capacity during these windows.
- Use the Logistic model to estimate probability of delay and scale capacity when delay risk is high.
- Plan for weather-related demand changes (e.g., rain, extreme heat).

### 5.3 Better Training for Delivery Staff
- If **Delivery_Person_Experience** is available, focus training on less experienced staff.
- Use **Weather** and **Vehicle_Type** to train for adverse conditions and vehicle handling.
- Monitor performance by traffic conditions and time of day to identify training gaps.

---

## 6. Deliverables Summary

| Deliverable | Description |
|-------------|-------------|
| **Jupyter Notebook** | `phase1_data_preprocessing.ipynb` — Full pipeline: preprocessing, EDA, feature engineering, model training, evaluation, and visualizations |
| **Processed Data** | `Food_Delivery_Time_Prediction_processed.csv` — Preprocessed dataset for modeling |
| **Scaler** | `scaler_phase1.pkl` — Fitted StandardScaler for inference |
| **Final Report** | `Final_Report.md` — This document |

---

## 7. How to Run

1. Place `Food_Delivery_Time_Prediction.csv` in the project folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Open and run `phase1_data_preprocessing.ipynb` (Run All or run cells in order).
4. Outputs: processed CSV, scaler, and all visualizations in the notebook.

---

*Report generated for the Food Delivery Time Prediction project.*
