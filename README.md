# Solar-Power-Generation-Prediction-Using-Machine-Learning-Models
This model is a machine learning-based solar power generation prediction system that uses historical weather and energy data to forecast solar output. It applies techniques like MLR, Lasso, Ridge, DT, GB, XGB, RF, SVR and Artificial Neural Networks to improve accuracy, helping optimize energy planning and grid management

## Project Overview

This project focuses on predicting DC power generation in a solar power plant based on weather conditions and time-based features. It involves a complete machine learning pipeline, from exploratory data analysis and preprocessing to building and evaluating multiple regression models.

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

- **Data Loading & Inspection**: Loading and initial inspection of generation and weather sensor data.
- **Data Preprocessing**: Handling datetime conversions, merging datasets, resampling to a consistent frequency (15-minute intervals), and imputing missing values.
- **Exploratory Data Analysis (EDA)**: Visualizing correlations, distributions, and relationships between features and the target variable.
- **Feature Engineering**: Extracting time-based features (hour, minute, day of week) from datetime objects.
- **Data Splitting & Scaling**: Dividing data into training and testing sets and applying `StandardScaler` to features.
- **Model Training**: Implementing and training various regression models.
- **Model Evaluation**: Assessing model performance using R-squared, MAE, MSE, and RMSE, along with actual vs. predicted plots and residual analysis.
- **Model Comparison**: A comparative analysis of all trained models to identify the best performer.

## Data Source

The data used in this project is sourced from the UCI Machine Learning Repository and consists of two main datasets:

1.  **`Generation_Data.csv`**: Contains information about power generation (DC and AC power, daily yield, total yield) from the solar plant.
2.  **`Weather_Sensor_Data.csv`**: Contains environmental data such as ambient temperature, module temperature, and irradiation.

Both datasets are collected over a period from May 15, 2020, to June 17, 2020, with varying granularities that are harmonized during preprocessing.

## Methodology

The project follows a standard machine learning workflow:

1.  **Initial Setup**: Import necessary libraries.
2.  **Data Loading**: Load `Generation_Data.csv` and `Weather_Sensor_Data.csv`.
3.  **Data Inspection & Preprocessing**: Convert `DATE_TIME` columns to datetime objects, merge the two datasets on `Date_Time`.
4.  **Resampling**: Resample the merged data to a 15-minute frequency to create a consistent time series.
5.  **Missing Value Handling**: Use forward-fill imputation to address any missing values after resampling.
6.  **Data Visualization**: Generate correlation heatmaps, pair plots, box plots for outlier detection, and histograms for feature distribution analysis.
7.  **Feature Engineering**: Extract 'hour', 'minute', and 'day_of_week' from the `Date_Time` column.
8.  **Data Splitting**: Split the dataset into training (80%) and testing (20%) sets.
9.  **Data Scaling**: Apply `StandardScaler` to the features of both training and testing sets.
10. **Model Training & Evaluation**: Train and evaluate a suite of regression models, comparing their performance.

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
