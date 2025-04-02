# Titanic Survival Prediction using Machine Learning

## Overview
This project implements a Machine Learning model to predict the survival of passengers on the Titanic based on various features such as age, sex, class, number of siblings/spouses aboard, fare, and embarked location. The dataset used for training and evaluation contains multiple attributes related to Titanic passengers.

## Dataset
The dataset (`titanic.csv`) consists of 891 records with the following features:
- `PassengerId` (unique identifier)
- `Survived` (target variable: 1 = survived, 0 = not survived)
- `Pclass` (ticket class: 1st, 2nd, 3rd)
- `Name` (passenger name)
- `Sex` (male or female)
- `Age` (age of passenger)
- `SibSp` (number of siblings/spouses aboard)
- `Parch` (number of parents/children aboard)
- `Ticket` (ticket number)
- `Fare` (passenger fare)
- `Cabin` (cabin number, if available)
- `Embarked` (port of embarkation: C = Cherbourg, Q = Queenstown, S = Southampton)

## Steps Involved
1. **Load the dataset**: The dataset is read using Pandas.
2. **Exploratory Data Analysis (EDA)**:
   - Display dataset information and summary statistics.
   - Visualize survival distribution using count plots.
   - Analyze relationships between survival and other features.
3. **Preprocessing**:
   - Handle missing values:
     - Replace missing values in `Age` with the median age.
     - Drop the `Cabin` column due to too many missing values.
     - Replace missing values in `Embarked` with the mode.
   - Convert categorical variables (`Sex` and `Embarked`) into numerical values.
4. **Feature Selection**:
   - Drop non-relevant features (`PassengerId`, `Name`, `Ticket`).
   - Select features: `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, and `Embarked`.
5. **Train-Test Split**:
   - Split the dataset into training and testing sets (80-20 ratio).
6. **Model Training**:
   - Use Logistic Regression as the baseline model.
   - Experiment with other models like Decision Trees and Random Forest.
7. **Prediction & Evaluation**:
   - Predictions are made using the trained models.
   - Evaluation metrics used:
     - Accuracy Score
       
## Code Execution
Run the Python script using:
```bash
python titanic_survival_prediction.py
```

## Results
- **Logistic Regression Accuracy**: 78.21%

## Visualization
Various plots were generated for data analysis:
- **Survival Count**: Bar plot of survived vs. non-survived passengers.
- **Sex-based Survival**: Bar plot comparing male and female survival rates.
- **Class-based Survival**: Bar plot showing survival rates for different ticket classes.
- **Age Distribution**: Histogram showing the age distribution of passengers.
- **Correlation Heatmap**: Displaying feature correlations.

## Future Improvements
- Try advanced classification models like XGBoost or Neural Networks.
- Implement feature engineering to create new meaningful variables.
- Perform hyperparameter tuning for all models.
- Utilize cross-validation to improve generalization.
