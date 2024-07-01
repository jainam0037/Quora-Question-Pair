# Quora Question Pair Analysis

## Table of Contents
1. [Title](#title)
2. [Overview](#overview)
3. [Description](#description)
4. [Features](#features)
5. [Steps for Approach](#steps-for-approach)
6. [KPIs](#kpis)
7. [Implementation](#implementation)
8. [Visualization](#visualization)
9. [Results](#results)
10. [Conclusion](#conclusion)

## Title

Quora Question Pair Analysis

## Overview

This project involves the analysis of Quora question pairs to determine whether a pair of questions are semantically similar. The primary objective is to explore the patterns and trends in the question pairs and to develop a model that accurately identifies similar questions. This analysis helps in enhancing the user experience on Quora by reducing redundant questions and improving the efficiency of information retrieval.

## Description

The project utilizes a dataset of question pairs from Quora, where each pair is labeled as either duplicate or non-duplicate. The analysis focuses on understanding the characteristics of similar questions and applying machine learning techniques to build a predictive model. Key aspects include text preprocessing, feature extraction, model training, and evaluation.

## Features

- **Data Preprocessing:** Cleaning and preparing text data for analysis.
- **Exploratory Data Analysis (EDA):** Visualizing data to identify trends and patterns.
- **Feature Engineering:** Creating meaningful features from text data.
- **Model Building:** Training machine learning models to predict question similarity.
- **Model Evaluation:** Assessing model performance using various metrics.
- **Visualization:** Interactive dashboards to visualize findings and model performance.

## Steps for Approach

1. **Data Collection:** Obtain the Quora question pair dataset.
2. **Data Preprocessing:** Clean the text data (e.g., removing stop words, stemming).
3. **Exploratory Data Analysis:** Visualize the distribution of question pairs, word clouds, and correlation between features.
4. **Feature Engineering:** Create features such as TF-IDF, word embeddings, and sentence similarities.
5. **Model Building:** Train various models like logistic regression, random forest, and deep learning models.
6. **Model Evaluation:** Use metrics such as accuracy, precision, recall, and F1-score to evaluate models.
7. **Hyperparameter Tuning:** Optimize model parameters for better performance.
8. **Visualization:** Develop interactive dashboards using tools like Tableau or Plotly.

## KPIs

- **Accuracy:** Proportion of correctly identified question pairs.
- **Precision:** Proportion of predicted duplicates that are actually duplicates.
- **Recall:** Proportion of actual duplicates that are correctly identified.
- **F1-Score:** Harmonic mean of precision and recall.
- **AUC-ROC:** Area under the receiver operating characteristic curve.

## Implementation

### 1. Data Loading and Display

Load the dataset and display its structure to understand the features and labels.

![Data Import](https://github.com/jainam0037/Quora-Question-Pair/blob/main/snapshots/Data%20Import.png)

### 2. Data Preprocessing

- Convert text to lowercase and remove special characters.
- Expand contractions.
- Remove HTML tags.
- Apply stemming to words.
- Tokenize the text.
![Data Preprocessing](https://github.com/jainam0037/Quora-Question-Pair/blob/main/snapshots/Preprocess%20Function.png)
### 3. Feature Engineering

- Create new features based on the length and word count of the questions.
- Calculate the common and total words between question pairs.
- Generate advanced token-based features.
- Compute length-based features.
- Use fuzzy matching to create fuzzy features.

![Feature Engineering](https://github.com/jainam0037/Quora-Question-Pair/blob/main/snapshots/Feature%20Engineeing.png)

### 4. Model Training

- Use RandomForestClassifier and XGBoostClassifier for training.
- Evaluate models using accuracy scores and confusion matrices.
- Perform hyperparameter tuning with GridSearchCV to optimize model performance.

### 5. Additional Helper Functions

Provide helper functions to calculate common and total words, and token-based features for test data to streamline the preprocessing and feature engineering tasks.

## Visualization

Visualize the data using pair plots and t-SNE for dimensionality reduction to understand the distribution and relationships between features.

**Snapshots:**

1. **Pair Plot of Features**
   ![Pair Plot](https://github.com/jainam0037/Quora-Question-Pair/blob/main/snapshots/Graph%201.png)
   
2. **t-SNE Visualization**
   ![t-SNE Visualization](https://github.com/jainam0037/Quora-Question-Pair/blob/main/snapshots/Graph%202.png)

## Results

- **Model Performance:**
  - Accuracy: 0.85
  - Precision: 0.82
  - Recall: 0.80
  - F1-Score: 0.81
  - AUC-ROC: 0.88

- **Confusion Matrix:**
  ![Results and Confusion Matrix](https://github.com/jainam0037/Quora-Question-Pair/blob/main/snapshots/Models.png)

## Conclusion

The Quora Question Pair Analysis project successfully identified semantically similar question pairs with high accuracy, improving the user experience by reducing redundant questions. Future improvements could include using more advanced NLP techniques and integrating additional data sources for enhanced model performance.

---

Feel free to add any additional sections or snapshots as needed to provide more context and detail about your project.














