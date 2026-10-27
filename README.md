# Titanic Survival Prediction using Logistic Regression

## 📌 Project Overview

This project predicts whether a passenger would survive the Titanic disaster using a Machine Learning classification algorithm called Logistic Regression.

The project includes the complete Machine Learning workflow, from data preprocessing and exploratory data analysis to model training, evaluation, and deployment using Flask.

## 🎯 Objective

The main objective of this project is to build a Logistic Regression model that predicts passenger survival based on information such as:

- Passenger class
- Gender
- Age
- Number of siblings/spouses
- Number of parents/children
- Ticket fare
- Port of embarkation

## 📊 Dataset

The project uses the Titanic dataset from Kaggle.

The target variable is:

- `Survived = 0` → Did not survive
- `Survived = 1` → Survived

### Features Used

| Feature | Description |
|---|---|
| Pclass | Passenger class |
| Sex | Gender |
| Age | Passenger age |
| SibSp | Number of siblings/spouses |
| Parch | Number of parents/children |
| Fare | Ticket fare |
| Embarked | Port of embarkation |

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Logistic Regression
- Flask
- HTML
- CSS
- Google Colab
- VS Code

## 🤖 Machine Learning Workflow

The project follows these steps:

1. Load the Titanic dataset
2. Understand the dataset
3. Handle missing values
4. Perform exploratory data analysis
5. Select relevant features
6. Encode categorical variables
7. Split data into training and testing sets
8. Apply feature scaling
9. Train Logistic Regression model
10. Make predictions
11. Evaluate the model
12. Save the trained model using Pickle
13. Deploy the model using Flask

## 🔄 Data Preprocessing

Missing values in the dataset were handled before training.

The `Age` column was filled using the median value.

The `Embarked` column was filled using the most frequent value.

Categorical variables such as `Sex` and `Embarked` were converted into numerical values using one-hot encoding.

Feature scaling was performed using `StandardScaler`.

## 📈 Model

The Machine Learning algorithm used in this project is:

**Logistic Regression**

Logistic Regression is suitable for this project because the target variable is binary:

- 0 → Not Survived
- 1 → Survived

## 📊 Model Evaluation

The model was evaluated using:

- Accuracy
- Confusion Matrix
- Precision
- Recall
- F1-Score
- Classification Report

## 🌐 Flask Web Application

The trained Logistic Regression model is deployed using Flask.

The web application allows users to enter passenger information such as:

- Passenger class
- Gender
- Age
- Siblings/spouses
- Parents/children
- Ticket fare
- Port of embarkation

After submitting the information, the application predicts whether the passenger is likely to survive and displays the survival probability.

## 📁 Project Structure

```text
titanic-survival-logistic-regression/
│
├── app.py
├── titanic_model.pkl
├── titanic_scaler.pkl
├── train.csv
├── Titanic_Logistic_Regression.ipynb
│
├── templates/
│   └── index.html
│
└── README.md
