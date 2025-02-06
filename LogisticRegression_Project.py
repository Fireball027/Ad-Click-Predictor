import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

ad_data = pd.read_csv('advertising.csv')

print(ad_data.head())
print(ad_data.info())
print(ad_data.describe())


# Exploratory Data Analysis (EDA)
# Create a histogram of the Age
plt.figure(figsize=(8, 5))
sns.histplot(ad_data['Age'], bins=30, kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Create a jointplot showing Area Income versus Age
sns.jointplot(x='Age', y='Area Income', data=ad_data)
plt.show()

# Create a jointplot showing the KDE distribution of Daily Time Spent on site versus Age
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, kind='kde', color='red')
plt.show()

# Create a jointplot of Daily Time Spent on site versus Daily Internet Usage
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data)
plt.show()

# Create a pairplot with the hue defined by the Clicked on Ad column feature
sns.pairplot(ad_data, hue='Clicked on Ad')
plt.show()


# Logistic Regression
# Split the data into training set and testing set
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Initialize and train the Logistic Regression model
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)


# Model Evaluation
# Make predictions on the test set
predictions = logmodel.predict(X_test)

# Print Classification Report and Confusion Matrix
print("\nClassification Report:\n", classification_report(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
