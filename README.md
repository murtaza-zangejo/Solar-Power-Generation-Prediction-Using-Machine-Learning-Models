## Solar-Power-Generation-Prediction-Using-Machine-Learning-Models
This model is a machine learning-based solar power generation prediction system that uses historical weather and energy data to forecast solar output. It applies techniques like MLR, Lasso, Ridge, DT, GB, XGB, RF, SVR and Artificial Neural Networks to improve accuracy, helping optimize energy planning and grid management

## Project Overview

**📌 Overview**
This project develops a predictive analytics system for solar energy generation using machine learning techniques.
It processes real-world solar plant data and evaluates multiple regression models to accurately forecast DC power output based on weather conditions and time-based patterns.

The system is designed following industry-grade ML pipeline practices, including preprocessing, feature engineering, model benchmarking, and evaluation.

**🎯 Objective**

To build a robust machine learning model that predicts solar power generation using:

Weather conditions (irradiance, temperature)
Time-series patterns (hour, day of week)
Historical energy production data

## Table of Contents

- [Features](#features)
- [Data Source](#data-source)
- [Methodology](#methodology)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Setup and Usage](#setup-and-usage)
- [Future Work](#future-work)
- [License](#license)

## Features

- **Data Loading & Inspection**: For  this project we are using Generation and weather data of a solar power plant which is downloaded from Kaggle https://www.kaggle.com/datasets/anikannal/solar-power-generation-data. We'll load the `Generation_Data.csv` and `Weather_Sensor_Data.csv` files into pandas DataFrames. We'll inspect the data types, check for date ranges, and convert date columns to datetime objects. We'll also merge the two datasets.
- **Data Preprocessing**: Handling datetime conversions, merging datasets, resampling to a consistent frequency (15-minute intervals), and imputing missing values. To ensure consistent time series and handle potential misalignments or missing intervals, we'll resample the data to a 15-minute frequency, taking the mean of values within each interval. We'll check for any missing values after resampling and impute them using forward-fill, which is a common method for time-series data.
- **Exploratory Data Analysis (EDA)**: Visualizing correlations, distributions, and relationships between features and the target variable. Visualize the correlation between all numerical features and the target variable (DC_POWER) using a heatmap. This helps understand the relationships between variables.
- **Feature Engineering**: Extract relevant time-based features such as 'hour', 'minute', and 'day of week' from the Date_Time column, as these can influence solar power generation. We'll define our independent variables (X) which are the features we'll use for prediction, and our dependent variable (y), which is the DC_POWER we want to predict.
- **Data Splitting & Scaling**: Dividing data into training and testing sets and applying `StandardScaler` to features. We'll split our dataset into a training set (80%) and a testing set (20%) to evaluate the model's performance on unseen data. Scaling features is often crucial for models that are sensitive to the magnitude of features (e.g., neural networks or models using regularization). We will scale our features using StandardScaler, fitting it on the training data and transforming both training and test data.
- **🤖 Machine Learning Models Training**: In this  project we have trained various regression algorithms, 
The following regression models were implemented:
Linear Regression.
Ridge Regression.
Lasso Regression.
KNN Regressor.
Decision Tree Regressor.
Random Forest Regressor.
Gradient Boosting Regressor.
XGBoost Regressor.
Support Vector Regressor (SVR).
- **Model Evaluation**: Assessing model performance using R-squared, MAE, MSE, and RMSE, along with actual vs. predicted plots and residual analysis.
- **Model Comparison**: A comparative analysis of all trained models to identify the best performer. And compared their results, so that Gradient Boosting won by achieving highest r2-score of 0.9946, followed by random forest and xgboost with r2-score of 0.9944 and 0.9936 respectively

## Data Source

The data used in this project is sourced from Kaggle datasets and consists of two main datasets:

1.  **`Generation_Data.csv`**: Contains information about power generation (DC and AC power, daily yield, total yield) from the solar plant.
2.  **`Weather_Sensor_Data.csv`**: Contains environmental data such as ambient temperature, module temperature, and irradiation.

Both datasets are collected over a period from May 15, 2020, to June 17, 2020, with varying granularities that are harmonized during preprocessing.

**⚙️ Tech Stack**
Language: Python
Libraries: Pandas, NumPy, Scikit-learn, XGBoost
Visualization: Matplotlib, Seaborn
Environment: Jupyter Notebook, Colab

## Methodology

The project follows a standard machine learning workflow:
🧠 ML Pipeline Architecture
Data Collection
      ↓
Data Cleaning & Preprocessing
      ↓
Time-Series Alignment (15-min resampling)
      ↓
Data Visualization
      ↓
Feature Engineering
      ↓
Train/Test Split
      ↓
Feature Scaling (StandardScaler) and Data Splitting
      ↓
Model Training (Multiple Regressors)
      ↓
Model Evaluation & Comparison
      ↓
Best Model Selection

1. **Data Collection**
    **Initial Setup**: Import necessary libraries.
    **Data Loading**: Load `Generation_Data.csv` and `Weather_Sensor_Data.csv`.
2. **Data Cleaning & Preprocessing**: Convert `DATE_TIME` columns to datetime objects, merge the two datasets on `Date_Time`.
3. **Time-Series Alignment (15-min resampling)**
    **Resampling**: Resample the merged data to a 15-minute frequency to create a consistent time series.
    **Missing Value Handling**: Use forward-fill imputation to address any missing values after resampling.
4. **Data Visualization**: Generate correlation heatmaps, pair plots, box plots for outlier detection, and histograms for feature distribution analysis.<img width="686" height="335" alt="image" src="https://github.com/user-attachments/assets/d120da13-2744-432e-9f26-37e737821954" />

5. **Feature Scaling (StandardScaler) and Data Splitting**
    **Feature Engineering**: Extract 'hour', 'minute', and 'day_of_week' from the `Date_Time` column.
    **Data Splitting**: Split the dataset into training (80%) and testing (20%) sets.
    **Data Scaling**: Apply `StandardScaler` to the features of both training and testing sets.
6. **Model Training & Evaluation**: Train and evaluate a suite of regression models, comparing their performance.

## Machine Learning Models

The following regression models are implemented and evaluated:

-   Linear Regression
-   Ridge Regression
-   Lasso Regression
-   K-Nearest Neighbors (KNN) Regressor
-   Decision Tree Regressor
-   Random Forest Regressor
-   Gradient Boosting Regressor
-   XGBoost Regressor
-   Support Vector Regressor (SVR)

## Results

The models are evaluated based on R-squared, Mean Absolute Error (MAE), Mean Squared Error (MSE), and Root Mean Squared Error (RMSE). A bar chart visualizes the R-squared scores of all models for easy comparison. Initial results indicate that **Gradient Boosting** and **Random Forest** models generally perform best, achieving high R-squared scores, suggesting they capture the variance in DC power generation effectively.
<img width="438" height="290" alt="image" src="https://github.com/user-attachments/assets/190c8b3d-8096-4fe0-8a72-560ad94e20b7" />

<img width="170" height="254" alt="image" src="https://github.com/user-attachments/assets/d248cbc8-b383-49fe-b668-9554f1d4b451" />
<img width="530" height="293" alt="image" src="https://github.com/user-attachments/assets/83627e14-e37d-4aa6-963e-ef64fa659022" />



## Setup and Usage

To run this project, follow these steps:

1.  **Clone the repository** (once it's on GitHub):
    ```bash
    git clone <repository-url>
    cd solar-power-prediction
    ```

2.  **Prepare the Data**:
    Ensure `Generation_Data.csv` and `Weather_Sensor_Data.csv` are placed in the `data/` directory (or directly in the same directory as the notebook if running locally without the full structure).

3.  **Install Dependencies**:
    It is recommended to use a virtual environment. The required libraries are:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```
    (A `requirements.txt` file will contain a full list of dependencies for easy installation: `pip install -r requirements.txt`)

4.  **Run the Notebook**:
    Open `solar_power_prediction.ipynb` in a Jupyter environment (like Google Colab, Jupyter Notebook, or JupyterLab) and run all cells sequentially.

## Future Work

-   **Hyperparameter Tuning**: Optimize the parameters of the best-performing models (e.g., Gradient Boosting, Random Forest) using techniques like GridSearchCV or RandomizedSearchCV.
-   **Time Series Cross-Validation**: Implement more robust cross-validation strategies suitable for time series data.
-   **Deep Learning Models**: Explore recurrent neural networks (RNNs) or LSTMs for time series prediction.
-   **Deployment**: Develop a simple web application or API to serve predictions from the trained model.
-   **Real-time Data Integration**: Integrate with live weather data APIs for real-time predictions.

## License

This project is open-source 
