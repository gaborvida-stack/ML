## Logistic Regression Classifier with Scikit-Learn
This project is a simple implementation of a logistic regression classifier using the Scikit-Learn library. The script allows users to choose between two datasets — the Iris dataset and the Wine dataset — and trains a logistic regression model on the selected dataset. After training, it evaluates the model using accuracy score, classification report, and confusion matrix.

## Features
  - Load either the Iris or Wine dataset
  - Split the dataset into training and test sets
  - Standardize the features for better model performance
  - Train a Logistic Regression model
  - Evaluate the model using accuracy, classification report, and confusion matrix

## Installation
  - Clone the repository or download the project files:
  ```bash
  git clone https://github.com/yourusername/logistic-regression-classifier.git
  ```
  ```bash
  cd logistic-regression-classifier
  ```

  - Install the required dependencies:
  ```
  pip install -r requirements.txt
  ```
  
## Usage

  ```bash
  python logistic_regression.py
  ```

  - You will be prompted to select a dataset: either iris or wine. Based on your choice, the program will:
    - Load the dataset.
    - Split the data into training and test sets.
    - Train the logistic regression model on the training set.
    - Evaluate the model's performance on the test set using metrics like accuracy, classification report, and confusion matrix.

## Example
```python
Choose a dataset (iris/wine): iris
Accuracy: 96.67%
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      0.93      0.97        15
           2       0.91      1.00      0.95         5

    accuracy                           0.97        30
   macro avg       0.97      0.98      0.97        30
weighted avg       0.97      0.97      0.97        30

Confusion Matrix:
 [[10  0  0]
  [ 0 14  1]
  [ 0  0  5]]
```
## Datasets
  - Iris Dataset: A dataset containing 150 samples of iris flowers with 4 features each (sepal length, sepal width, petal length, and petal width), classified into 3 species.
  - Wine Dataset: A dataset containing 178 samples of wine with 13 chemical properties used to classify the wine into 3 categories.
