## Overview

The **Ad Click Prediction Project** aims to analyze user behavior and predict whether an individual will click on an advertisement. Using exploratory data analysis (EDA) and machine learning techniques, this project provides insights into how factors such as age, internet usage, and time spent on a website impact ad engagement.

---

## Key Features

- **Data Preprocessing**: Cleans and prepares the dataset for analysis.
- **Exploratory Data Analysis (EDA)**: Visualizes relationships between different features.
- **Logistic Regression Model**: Trains a predictive model to classify users as ad clickers or non-clickers.
- **Model Evaluation**: Assesses model performance using classification metrics.

---

## Project Files

### 1. `advertising.csv`
This dataset contains user data, including:
- **Daily Time Spent on Site**: Time a user spends on the website.
- **Age**: User’s age.
- **Area Income**: Average income of the user’s geographic area.
- **Daily Internet Usage**: Amount of time spent on the internet daily.
- **Male**: Binary indicator of gender (1 = Male, 0 = Female).
- **Clicked on Ad**: Target variable indicating if the user clicked on an ad (1 = Yes, 0 = No).

### 2. `ad_click_prediction.py`
This script performs EDA, trains a logistic regression model, and evaluates its performance.

#### Key Components:

- **Exploratory Data Analysis (EDA)**:
  - Histograms and jointplots for feature distributions.
  - KDE plots for relationship insights.
  - Pairplots with class distinction.

- **Logistic Regression Model**:
  - Splits data into training and testing sets.
  - Fits a logistic regression model.
  - Predicts and evaluates model performance.

#### Example Code:
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv('advertising.csv')

# Train-test split
X = data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = data['Clicked on Ad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
```

---

## How to Run the Project

### Step 1: Install Dependencies
Ensure required libraries are installed:
```bash
pip install pandas seaborn matplotlib scikit-learn
```

### Step 2: Run the Script
Execute the main script:
```bash
python ad_click_prediction.py
```

### Step 3: View Insights
- Model performance metrics (Classification Report, Confusion Matrix)
- Visualizations of user behavior trends
- Predicted vs Actual Ad Click outcomes

---

## Future Enhancements

- **Feature Engineering**: Create additional variables for better predictions.
- **Advanced Models**: Experiment with decision trees and neural networks.
- **Hyperparameter Tuning**: Optimize logistic regression parameters.
- **Web App Integration**: Deploy as an interactive web tool for marketing teams.

---

## Conclusion

The **Ad Click Prediction Project** provides valuable insights into online user behavior and enhances targeted advertising strategies. By leveraging logistic regression and data visualization, it serves as a powerful tool for predicting ad engagement.

---

**Happy Predicting!**
