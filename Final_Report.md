# Food Delivery Time Prediction — Final Report

## Project Overview

This project predicts food delivery times using customer location, restaurant location, weather, traffic, and related factors. It combines **data preprocessing**, **exploratory data analysis**, **feature engineering**, and **predictive modeling** (Linear Regression and Logistic Regression) to support operational decisions.

---

## 1. Dataset Description

### Source
- **File:** `Food_Delivery_Time_Prediction.csv`
- **Target variable:** `Delivery_Time` (continuous, in minutes or similar units)

The dataset contains individual food delivery orders. Each row represents one completed delivery with information about the **order**, the **route**, and the **context** (weather, traffic, time of day, etc.). The goal is to understand which factors drive delivery duration and to build models that can predict or classify delivery performance.

### Core Features Used
| Feature | Type | Description |
|---------|------|-------------|
| Distance | Numeric | Distance between restaurant and delivery location (km, computed via Haversine if not provided) |
| Delivery_Time | Numeric | Actual delivery duration (target) |
| Order_Cost | Numeric | Cost of the order (bill amount) |
| Weather_Conditions | Categorical | Weather at delivery time (e.g., Clear, Rainy, Stormy) |
| Traffic_Conditions | Categorical | Traffic level (e.g., Low, Medium, High) |
| Vehicle_Type | Categorical | Delivery vehicle used (e.g., Bike, Scooter, Car) |
| Order_Hour | Numeric | Hour of order (0–23) extracted from order timestamp |
| Rush_Hour | Binary | 1 if order placed in peak windows (11–14, 18–21), else 0 |

### Optional / Contextual Columns (if present)
- Restaurant and customer latitude/longitude (for Haversine distance)
- Raw order time / date columns (used to derive `Order_Hour` and `Rush_Hour`)
- Operational features such as `Order_Priority`, `Delivery_Person_Experience`, etc., which can further refine the models.

---

## 2. Preprocessing Steps

### 2.1 Data Import and Cleaning
- Loaded CSV and normalized column names (handled common variants and typos).
- **Missing values:** Imputed numeric columns with median if &lt;5% missing; dropped rows if ≥5% missing; imputed categorical columns with mode.
- Dropped any remaining rows with NaN.

### 2.2 Exploratory Data Analysis
- **Descriptive statistics:** For each numeric feature, we computed **mean**, **median**, **mode**, **variance**, and **standard deviation**. The **mode** (most frequent value) was shown explicitly in a separate table to make it clear which value occurs most often for each variable (e.g., common order cost bands or typical delivery durations).
- **Correlation analysis:** We calculated correlations between each numeric feature and `Delivery_Time`, visualized via a **correlation heatmap** and a **bar plot**. Features with larger absolute correlation values are more strongly associated with delivery time (e.g., Distance and Rush_Hour typically show strong positive relationships, while some cost-related variables may have weaker effects).
- **Outlier detection:** Boxplots were used to identify extreme values in numeric features. Instead of deleting rows, numeric outliers were **capped at 1.5×IQR** (interquartile range), which reduces the impact of extreme values while preserving data volume.

### 2.3 Feature Engineering
- **Distance (Haversine):** If no `Distance` column existed, we computed the great-circle distance in kilometers from restaurant to customer using latitude and longitude columns (Haversine formula). This converts raw geo-coordinates into a single, interpretable measure directly used by the models.
- **Time-based features:** We extracted `Order_Hour` from timestamp-like columns and created `Rush_Hour` (1 for 11–14 or 18–21, else 0). These features capture daily demand patterns and congestion, which are important drivers of delivery time.

### 2.4 Data Transformation
- **Categorical encoding:** One-hot encoding for Weather, Traffic, Vehicle Type (and similar columns), with `drop_first=True`.
- **Standardization:** StandardScaler applied to numeric features (excluding the target) for zero mean and unit variance.
- **Outputs:** `Food_Delivery_Time_Prediction_processed.csv`, `scaler_phase1.pkl`.

---

## 3. Model Evaluation and Comparisons

### 3.1 Linear Regression (Delivery Time Prediction)

**Objective:** Predict continuous `Delivery_Time` from features such as Distance, time-based features, and encoded traffic/weather/vehicle variables.

**Setup:**
- Train–test split: 80% train, 20% test (`random_state=42`).
- Features: All numeric columns except the target (including standardized Distance, Order_Cost, Order_Hour, Rush_Hour, and one‑hot–encoded categorical dummies).
- Target: `Delivery_Time` (original scale, not standardized).

**Metrics:**
| Metric | Description |
|--------|-------------|
| MSE | Mean Squared Error — average squared prediction error |
| MAE | Mean Absolute Error — average absolute error |
| R² | Coefficient of determination — proportion of variance explained |

**Visualizations:**
- **Predicted vs Actual scatter plot:** Shows how close model predictions are to the true delivery times; points close to the diagonal line indicate accurate predictions.

**Feature importance (regression coefficients):**
- After standardizing numeric predictors, we examined the **Linear Regression coefficients**. Larger absolute coefficients indicate stronger influence on `Delivery_Time`.
- Typically, **Distance** and **Rush_Hour** have the largest positive coefficients: longer routes and peak-time orders significantly **increase** delivery time.
- Negative or small coefficients (e.g., for some vehicle-type dummies) suggest features with minimal or slightly reducing impact compared to the baseline.

---

### 3.2 Logistic Regression (Fast vs Delayed Classification)

**Objective:** Classify deliveries as **Fast** (1) or **Delayed** (0) using Traffic, Weather, vehicle, and time-based features.

**Setup:**
- Binary target: Fast = `Delivery_Time` ≤ median, Delayed = above median.
- Same 80/20 split and feature set as Linear Regression (but with a binary target).

**Metrics:**
| Metric | Description |
|--------|-------------|
| Accuracy | Proportion of correct predictions |
| Precision | Among predicted Fast, proportion actually Fast |
| Recall | Among actual Fast, proportion correctly predicted |
| F1-score | Harmonic mean of Precision and Recall |
| Confusion Matrix | True/False Positives and Negatives |

**Visualizations and interpretation:**
- **Confusion matrix heatmap:** Shows how many Fast/Delayed deliveries were correctly and incorrectly classified. High values along the diagonal and low off-diagonal counts indicate good performance.
- **ROC curve and AUC:** Summarize the trade‑off between true positive and false positive rates. A higher AUC means that the model discriminates well between Fast and Delayed deliveries across thresholds.

---

### 3.3 Model Comparison

| Model | Task | Primary Metrics |
|-------|------|-----------------|
| Linear Regression | Predict delivery time | MSE, MAE, R² |
| Logistic Regression | Classify Fast vs Delayed | Accuracy, Precision, Recall, F1 |

- **Linear Regression** is used for precise time estimates and planning.
- **Logistic Regression** is used for binary classification and risk assessment (e.g., likelihood of delay).

---

## 4. Data Visualizations and EDA Interpretation

The notebook includes the following plots **and their interpretations**:

1. **Histograms** — Distributions of Distance, Delivery_Time, and Order_Cost. These reveal whether variables are symmetric or skewed (for example, Delivery_Time is often right‑skewed, with many short deliveries and fewer very long ones).
2. **Missing value heatmap** — Shows where missing values occur; in this dataset, missingness was limited and handled via imputation, so no entire feature was dropped.
3. **Correlation heatmap** — Highlights which numeric features move together. Distance and Rush_Hour typically show strong positive correlation with Delivery_Time; weaker correlations for some cost-related features indicate limited impact on time.
4. **Correlation bar plot (with Delivery_Time)** — Orders features by absolute correlation, making it clear which predictors are most strongly associated with the target.
5. **Pair plot** — Visualizes pairwise relationships between Distance, Delivery_Time, Order_Cost, Order_Hour, and Rush_Hour. Points tend to rise with Distance and during Rush_Hour, confirming that these features drive longer delivery times.
6. **Boxplots** — Identify extreme values in Distance and Delivery_Time; after capping with the IQR rule, distributions become more robust and less dominated by outliers.
7. **Predicted vs Actual scatter plot (Linear Regression)** — Shows that predictions track the diagonal reasonably well, but with some spread for very long deliveries, suggesting that extreme conditions are harder to model perfectly.
8. **Confusion matrix heatmap (Logistic Regression)** — Indicates how many Fast/Delayed deliveries are correctly captured; a relatively high diagonal count shows the model’s practical usefulness.
9. **ROC curve** — Demonstrates that the Logistic model has discriminative power (AUC noticeably above 0.5), making it suitable as an early‑warning tool for potential delays.

---

## 5. Actionable Insights and Recommendations (Final Summary)

Based on the models and EDA, the following operational recommendations are proposed:

### 5.1 Optimizing Delivery Routes
- **Shorten effective distance:** Since Distance has one of the largest positive regression coefficients, prioritize route-planning algorithms that minimize travel distance while respecting traffic and road constraints.
- **Avoid high-traffic segments:** Use **Traffic_Conditions** (and, if available, live traffic feeds) to route drivers away from consistently congested roads, especially during Rush_Hour.
- **Weather‑aware routing and vehicle assignment:** In adverse **Weather_Conditions** (e.g., heavy rain), consider assigning more robust **Vehicle_Type** (e.g., cars instead of bikes) and choosing safer, slightly longer routes to maintain reliability.

### 5.2 Adjusting Staffing and Capacity During Peak Times
- **Align staffing with Rush_Hour:** Rush_Hour is strongly associated with longer delivery times. Increase the number of delivery partners and kitchen staff during 11–14 and 18–21 to keep predicted delays under control.
- **Use delay probability forecasts:** The Logistic Regression model outputs probabilities of a delivery being Delayed. Use these probabilities for **capacity planning** (e.g., opening extra packing counters when average predicted delay risk exceeds a threshold).
- **Weather and traffic surcharges:** In periods with both poor weather and heavy traffic, consider temporary adjustments (e.g., surge pay or dynamic batching) so that more drivers are available and deliveries can still be completed on time.

### 5.3 Training and Process Improvements
- **Targeted training for new or underperforming drivers:** If `Delivery_Person_Experience` (or similar metrics) is available, use model outputs to identify drivers whose deliveries are consistently predicted as Delayed and offer focused training on navigation and time management.
- **Best-practice playbooks per context:** Use patterns from Weather_Conditions, Traffic_Conditions, and Vehicle_Type to build scenario-based guidelines (e.g., how to drive and plan stops safely during heavy rain or major traffic events).
- **Continuous monitoring:** Regularly re-train the models with recent data so that changing patterns (new neighborhoods, new restaurants, or different customer behavior) are quickly captured and operational rules remain effective.

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
