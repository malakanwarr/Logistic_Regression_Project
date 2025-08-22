# Logistic Regression Project: Predicting Ad Clicks

## Project Overview
In this project, we work with a **fake advertising dataset** to predict whether an internet user will click on an advertisement based on their demographic and behavioral features. We apply **Logistic Regression**, a supervised machine learning algorithm, to classify user behavior.

The dataset includes the following features:

- **Daily Time Spent on Site**: Time spent on the website (in minutes)  
- **Age**: Age of the consumer (in years)  
- **Area Income**: Average income of the consumer's geographical area  
- **Daily Internet Usage**: Average daily time spent on the internet (in minutes)  
- **Ad Topic Line**: Headline of the advertisement  
- **City**: Consumer's city  
- **Male**: Whether the consumer is male (0 = No, 1 = Yes)  
- **Country**: Consumer's country  
- **Timestamp**: Time of interaction with the ad  
- **Clicked on Ad**: Target variable (0 = Did not click, 1 = Clicked)  

---

## Libraries Used
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```

## Data Exploration

We start by loading the dataset and exploring it:
```bash
ad_data = pd.read_csv('advertising.csv')
ad_data.head()
ad_data.info()
ad_data.describe()
```

We perform exploratory data analysis (EDA) using visualizations:

- Histogram of Age
- Jointplot of Area Income vs Age
- Jointplot (KDE) of Daily Time Spent on Site vs Age
- Jointplot of Daily Time Spent on Site vs Daily Internet Usage
- Pairplot colored by Clicked on Ad
```bash
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')

sns.jointplot(x='Age', y='Area Income', data=ad_data)
sns.jointplot(x='Age', y='Daily Time Spent on Site', data=ad_data, kind='kde', color='red')
sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data, color='green')
sns.pairplot(data=ad_data, hue='Clicked on Ad', palette='bwr')
```
## Data Preprocessing

We drop unnecessary columns (Ad Topic Line, City, Country, Timestamp) and split the dataset into features (X) and target (y).
```bash
X = ad_data.drop(['Clicked on Ad','Ad Topic Line','City','Country','Timestamp'], axis=1)
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```

## Logistic Regression Model
We train a Logistic Regression model on the training set:
```bash
lg = LogisticRegression()
lg.fit(X_train, y_train)
```

## Predictions and Evaluation

We predict on the test set and evaluate the model using a classification report and confusion matrix:
```
pred = lg.predict(X_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

```

## Results:
- Accuracy: 91%
- Precision & Recall: Balanced for both classes
- Confusion matrix shows good classification performance with few misclassifications.

## Conclusion

The Logistic Regression model successfully predicts whether a user will click on an ad based on their profile. Key insights:

- Users with more time spent on the site or higher daily internet usage are more likely to click.

- Age and income have observable trends that influence ad clicks.

This project demonstrates the complete workflow of a binary classification problem: EDA, preprocessing, model training, prediction, and evaluation.
