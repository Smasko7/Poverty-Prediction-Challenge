# 🌍 Poverty Prediction Online Competition Challenge

[![DrivenData](https://img.shields.io/badge/DrivenData-World%20Bank-blue)](https://www.drivendata.org/competitions/305/competition-worldbank-poverty/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)

This is a project aimed to take part in the World Bank Poverty Prediction Challenge of https://www.drivendata.org/. This challenge is an online data science competition on DrivenData. The main problem is to estimate individual and aggregate household consumption and total poverty rates, from limited survey data. More information on https://www.drivendata.org/competitions/305/competition-worldbank-poverty/.

<br>

---

## 📖 Table of Contents
* [📍 Problem Description](#-problem-description)
* [📁 Project Structure](#-project-structure)
* [⚙️ Workflow](#-workflow)
    * [1. Data Exploration & Preprocessing](#1-data-exploration-and-preprocessing)
    * [2. Model Training & Evaluation](#2-model-training-and-evaluation)
    * [3. Final Models Selection](#3-final-models-selection)
* [📊 Results](#-results-on-cross-validation-sets)
* [🚀 How to Run](#-how-to-run)
* [📜 License](#-license)

---


## 📍 Problem Description

For this challenge, the task is to use survey data to predict both the population’s poverty rate at various thresholds as well as household-level daily per capita consumption (`cons_ppp17`).
This challenge mimics a common challenge faced in real-time poverty monitoring in which older surveys fully capture household consumption data, but more recent surveys lack the detailed information needed to directly measure poverty rates. The process of inferring consumption data from less detailed surveys is called imputation. In this challenge, we will impute both household consumption and, with that information, the population-level poverty rate.


## 🎯 The Goal
The goal is to minimize the official competition metric: **wS-wMAPE**, which is a weighted sum of the household consumption and the population-level poverty rate.

---

<br>

## 📁 Project Structure

The repository is organized into 3 main Jupyter notebooks:

-   `Poverty_EDA_and_Preprocessing.ipynb`: This notebook covers all aspects of data exploration, cleaning, and feature engineering of the given datasets.
-   `Train_Models.ipynb`: This notebook focuses on model training, hyperparameter tuning, and evaluation using the competition's specific metrics.
-   `Final Models.ipynb`: This notebook derives the final models after fine-tuning and gets the final predictions on the competition's test set.


| Notebook | Focus |
| :--- | :--- |
| 🧹 **`Poverty_EDA_and_Preprocessing.ipynb`** | This notebook covers all aspects of data exploration, cleaning, and feature engineering of the given datasets. |
| 🏋️ **`Train_Models.ipynb`** | This notebook focuses on model training, hyperparameter tuning, and evaluation using the competition's specific metrics. |
| 🏆 **`Final_Models.ipynb`** | This notebook derives the final models after fine-tuning and gets the final predictions on the competition's test set. |

---

<br>

## ⚙️ Workflow

The project follows a comprehensive machine learning pipeline, from data preparation to model evaluation.

### 1. Data Exploration and Preprocessing

The `Poverty_EDA_and_Preprocessing.ipynb` notebook performs the following steps:

-   **🔍 Exploratory Data Analysis (EDA):**
An initial analysis of the dataset is conducted using `ydata_profiling` to understand feature distributions, correlations, and missing values. Specifically, the train dataset has 104234 samples and 88 features. It is divided in 3 separate surveys, which have been conducted in different years/countries.
The 50 out of the total 88 features have the form: "consumedXXXX", where each 4-digit number corresponds to a food type. The "consumedXXXX" are binary variables, that indicate wether a specific household had consumed each food type in the past 7 days. The other features correspond to:

<br>

    1. Identifiers & sampling information
    2. The weights in the dataset are population-expanded weights, meaning that they reflect the probability of sampling times the number of members in the household. These are used to convert household-level survey data to accurate population-level estimates.
    3. Welfare & expenditure information
    4. Demographics & household composition
    5. Education & employment
    6. Housing & utilities
    7. Geography

<br>

-   **🛠 Categorical Feature Encoding:**
Text-based categorical features are converted to numerical values based on the provided `feature_value_descriptions.csv`. However, the encoding of this file is not perfect for model training, because for some features there is increasing indexing of categorical values, without corresponding to actual ordinal encoding. Thus, further pre-processing should be made.

<br>

-   **🛠 Ordinal Encoding:** Specific features like `water_source`, `sanitation_source`, `educ_max` and 'dweltyp' are ordinally encoded to reflect a qualitative ranking (e.g., 'Piped water' is better than 'Surface water'). In other words, common sense is used to put the feature categories in semantic order.

<br>

-   **🛠 Missing Value Imputation:** Missing values are handled strategically. Not all features' missing values need the same treatment, so we handle them separately.
    -   `dweltyp` and consumption features (`consumedXXXX`) are filled with specific values. 
    -   Other features like `employed`, `educ_max`, and `share_secondary` are imputed using survey-specific modes and medians to prevent data leakage across different surveys.
    -   It is observed that some features' missing values are directly related to other features' missing values. Specifically, the households with missing values in 'sector1d' are those with either 'employed' = 0 or 'employed' = NaN. This means that if the job sector is missing in a household, either the 'employed' value is missing or the head of the household is unemployed. We fill the 'employed' missing values as the most frequent answer of the feature (0 or 1). Then, for the households with filled (imputed) 'employed' values 0, we also fill the corresponding 'sector1d' values with 0. But, for the households with imputed 'employed' values 1, we fill the corresponding 'sector1d' values with the most frequent answer (sector) of 'sector1d' feature.

<br>

-    **📈 Plot Target Distribution (`cons_ppp17`):**
  Target distributions for the whole train dataset and for the separate surveys' sub-sets are plotted. It is concluded that all distributions of cons_ppp17, in the 3 surveys and as a whole, are pretty similar to each other. Specifically, they follow an exponential (right skewed) distribution.

<br>

-   **🛠 Feature Engineering:**
    -    New features are created to capture more comprehensive information:
           1. `dietary_diversity`: A score representing the total count of different food items consumed.
           2. `grain_score` & `protein_score`: Aggregate scores for the consumption of essential food groups.
    -    Features that represent IDs (`survey_id`, `hhid`) or have zero variance (`com`) are dropped.

<br>

-   **🏆 Feature Ranking:** A fused feature ranking is generated by combining three methods to identify the most impactful features:
    1.  Correlation with the target variable (`cons_ppp17`). It is observed that there are no features with high correlation (e.g. > 0.5), which means that there is not a strong indicator, at least at first glance. The feature with the highest correlation with the target variable is the stratification feature (strata) and the expenditure on utilities (utl_exp_ppp17), which is completely expected and rational, given that cons_ppp17 and utl_exp_ppp17 variables measure directly related things (Per-capita daily expenditure and Expenditure on utilities). 
    2.  XGBoost feature importance.
    3.  Mutual Information.

We keep all the features in a ranked list, so that we can use it in wrapper methods.

<br>

-   **🛠 Data Scaling:** The final feature set is standardized using `StandardScaler` (Z-score normalization) and saved to `Preprocessed_train_data.csv`.

<br>

-   **🛠 Test Dataset Preprocessing** The same preprocessing procedure is applied to the original test set. The result is saved to `Preprocessed_test_data.csv`

**📁 Exported Files**
This notebook exports some files that resulted from the data preprocessing, which are used in the next notebooks:
1.  `Final_features_rank.csv`
2.  `Preprocessed_train_data.csv`
3.  `Preprocessed_test_data.csv`

---

<br>


### 2. Model Training and Evaluation

The `Train_Models.ipynb` notebook implements the modeling phase:

-   **📊Custom Competition Metric:** The function `calculate_poverty_rates` is implemented to calculate the official competition metric (wS-wMAPE), which is a weighted blend of the Mean Absolute Percentage Error (MAPE) for consumption predictions (10%) and poverty rate predictions (90%). The poverty rate predictions part is also weighted. These weights prioritize the poverty thresholds more the closer they are to the threshold corresponding to a 40% poverty rate. The competition metric is given below:

<img width="628" height="72" alt="Στιγμιότυπο οθόνης (248)" src="https://github.com/user-attachments/assets/cf8814b5-5243-4525-87c8-9fd5070f6536" />

More information in https://www.drivendata.org/competitions/305/competition-worldbank-poverty/page/965/#performance-metric.


<br>

-   **Cross-Validation Strategy:** A `GroupKFold` cross-validation approach is used, with `survey_id` as the grouping variable. This ensures that the model is trained on two surveys and validated on the third, promoting generalization to unseen survey data.


<br>

-   **Log Transformation:** The target variable (`cons_ppp17`) is log-transformed before training to handle its right-skewed distribution, with the inverse transformation applied to predictions.

<br>

-   **⚖️ Fit with weights:** The use of sample weights is a critical technical necessity for this problem, because each household weight represents a varying number of people in the total population. By fitting models with sample_weight=weights, we fundamentally alter the optimization objective: the model’s loss function is multiplied by these weights, forcing the model to prioritize "high-weight" households, that represent larger segments of the population. This alignment is vital because the competition’s 90/10 blended metric is itself a weighted calculation; training without weights would optimize for the dataset's internal distribution rather than the real-world population distribution the World Bank aims to measure.

<br>

-   **🧪 Model Exploration:**
  Several regression models are trained and evaluated using `GridSearchCV` to find the best hyperparameters:

    -  **1. Ridge Regression (Linear Baseline)**
        -   **Why Selected:** To establish a lower-bound performance metric and assess the linearity of the relationship between household assets and consumption.
        -   **Pros/Cons:** Fast and interpretable, but fails to capture "poverty traps" and non-linear interactions between variables.
        -   **Result:** Performance was expectedly low (12.2876) due to the non-linear nature of economic survey data.


    -  **2. Random Forest Regressor (Bagging Ensemble)**
        -   **Why Selected:** Known for robustness against outliers and its ability to handle high-dimensional feature spaces.
        -   **Pros/Cons:** Reduces variance through averaging (Bagging), but lacks the "surgical" precision of boosting to optimize specific thresholds.
        -   **Result:** Sub-optimal (13.1311). The level-wise growth was less effective than gradient-based methods for this distribution-heavy task.
     
    -  **3. TabNet (Attentive Deep Learning)**
        -   **Why Selected:** A specialized neural network designed to bring the benefits of deep learning to tabular data.
        -   **Pros/Cons:** Uses sequential attention to choose relevant features, but is highly sensitive to hyperparameters and requires more data to converge.
        -   **Result:** Strong performance (7.7381), but fell short of the top tree-based models on this specific dataset size.
     
    -  **4. XGBoost Regressor (Gradient Boosting)**
        -   **Why Selected:** The "Gold Standard" for tabular data, allowing for extreme flexibility and custom loss functions.
        -   **Pros/Cons:** Each tree specifically corrects the errors of the previous ones (Boosting).
        -   **Result:** **Winner (5.6561).** The ability to use sample weights to focus on the 40th percentile made it the most effective tool for the 90/10 challenge.

### Preliminary Results
The model **preliminary results** are depicted in the following plot:

<img width="1000" height="600" alt="Model_results" src="https://github.com/user-attachments/assets/c099293d-4b82-4906-a853-9adcbe910316" />

| Model | Selection Reason | Result |
| :--- | :--- | :--- |
| **Ridge Regression** | Linear baseline to assess relationship complexity. | 12.2876 (Poor) |
| **Random Forest** | Robustness against outliers and non-linearity. | 13.1311 (Poor) |
| **TabNet** | Modern deep learning designed for tabular data. | 7.7381 (Strong) |
| **XGBoost** | "Gold Standard" for GBDT tasks with high flexibility. | **5.6561 (Best)** |

---
  
<br>

### 3. Final Models Selection

As we have seen in Train_Models.ipynb, the most efficient model for this competition's regression task is xgboost, thus we are going to further fine-tune it.

Also, we observe the **necessity of 2 separate models for the 2 separate regression tasks (consumption per household and overall poverty rates)**. Only one model that first predicts the consumption and then derives the poverty rates is highly unlikely to perform well on both tasks.

As a result, in the `Final_Models.ipynb` notebook we: 

1.  First apply a grid search of xgboost model, with datasets' population weights taken into consideration as sample_weights, with the goal to minimize the weighted MAPE of poverty rates (secondary competetition metric). This metric plays a decisive role in the final competition metric, as it affects the 90% of the final metric.

2.  For the second task (predict consumption: 'consppp17'), we will try to fine-tune a CatBoost model, to yield even better results. It is mentioned that the datasets' weights should not be taken into account in fit() function this time, because predicting consumption is conducted per household and we are not interested in a total population metric here. This trick will result to even better performance.

<br>

-   **XGBoost to predict Poverty Rates and CatBoost to predict Consumption (`cons_ppp17`)**
It is noted that the poverty rates are derived from the consumption predictions, by the `calculate_poverty_rates` function. Thus, the xgboost model should inevitably predict, first, the consumption values and then extract the poverty rates. These poverty rates will be submitted in the competition. However, the consumption predictions (2nd task), that led to finding the poverty rates, will not be sumbitted, as the performance will be poor. For the second task, **Catboost without weights** is selected.

<br>

-  **🛠 Systematic Bias Correction (+0.005 Offset)**
During testing, a systematic under-prediction of poverty was identified. By applying a global **+0.005 (0.5%) refinement factor** to all poverty rate columns (excluding `survey_id`), the score improved dramatically. This successfully aligned the predicted cumulative distribution with the ground truth.


---
<br>


## 📊 Results on cross-validation sets

After extensive grid searches and evaluation, it was found that:

| Task | Model | Optimal Parameters | CV Score |
| :--- | :--- | :--- | :--- |
| **Poverty Rate** | XGBoost | `LR: 0.5, Max Depth: 12, Subsample: 0.55` | **1.1455** |
| **Consumption** | CatBoost | `LR: 0.1, Depth: 8, L2 leaf reg: 9, Iter: 1000` | **2.7689** |

<br>


## 🏆 Results on Test set (Open Competition Leaderboard)

The final strategy achieved **10th position** out of 300+ participants.
* **Final Competition Metric (wS-wMAPE):** `3.684`
* **Poverty Rates Weighted MAPE:** `0.910`
  
<img width="1253" height="883" alt="Στιγμιότυπο οθόνης (256)" src="https://github.com/user-attachments/assets/2940ced3-265b-4bf6-880e-e816ec20e233" />


<br>

The final output is the zip folder **`submission.zip`**, which is our official submission to the competition.

---

<br>



## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/smasko7/poverty-prediction-challenge.git
    cd poverty-prediction-challenge
    ```

2.  **Download Data:**
    Download the competition data from the [DrivenData competition page](https://www.drivendata.org/competitions/305/competition-worldbank-poverty/data/) and place the `.csv` files in the root directory.


3.  **Execute the Notebooks:**
    -   First, run `Poverty_EDA_and_Preprocessing.ipynb` to clean the data and generate the preprocessed files (`Preprocessed_train_data.csv`, `Preprocessed_test_data.csv`).
    -   Then, run `Train_Models.ipynb` to train the models and reproduce the preliminary results.
    -   Finally, run `Final Models.ipynb` to get the final models and export the submission zip folder.

---

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👥 Authors & Acknowledgments

* **Primary Contributor:** [smasko7](https://github.com/smasko7)
* **Competition Host:** [DrivenData](https://www.drivendata.org/)
* **Data Provider:** The World Bank Group

---

**Project Title:** World Bank Poverty Prediction Challenge  
**Final Competition Rank:** 10th Place 🏆
