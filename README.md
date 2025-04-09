**Heart Attack Prediction using Machine Learning**

📌 **Overview**
This project focuses on building a predictive model to identify individuals at risk of heart attacks using various machine learning algorithms. With heart disease being one of the leading causes of death worldwide, early detection through data-driven insights can support timely intervention and treatment. The project evaluates the performance of Logistic Regression, Random Forest, Naïve Bayes, and Support Vector Machine (SVM) classifiers on a heart disease dataset.

🧠 **Key Objectives**
Clean and preprocess a real-world dataset for heart disease prediction.
Explore relationships between health features using visualization tools and correlation matrices.
Train and evaluate multiple machine learning models to identify the most accurate and reliable predictor.
Compare model performance using accuracy, confusion matrices, and classification reports.
Highlight critical health indicators such as cholesterol, blood pressure, and chest pain type.

⚙️ **Technologies & Tools**
**Languages**: Python

**Libraries**:
**Data Manipulation:** pandas, numpy
**Visualization:**seaborn, matplotlib
**Machine Learning:** scikit-learn

**Models Used:**
Logistic Regression
Random Forest Classifier
Gaussian Naïve Bayes
Support Vector Machine (SVM)

**Evaluation Metrics:**
Accuracy Score
Confusion Matrix
Classification Report

📊 **Dataset**
The dataset contains multiple health-related features like age, sex, cholesterol, resting blood pressure, max heart rate, ST depression, and more.
The target column is the output label (0 = No Heart Disease, 1 = Heart Disease).

🔍** Data Exploration & Preprocessing**
Checked for null values and ensured correct data types.
Created a heatmap of correlations to understand feature importance.
Used pairplots and histograms for visual exploration of distributions.
Scaled features using StandardScaler before training ML models.

🧪 **Model Training & Evaluation**
Model	Accuracy	Notes
Logistic Regression	 ~0.79	Simple, interpretable model
Random Forest	       ~0.86	Highest accuracy, great with non-linear features
Naïve Bayes	         ~0.85	Lightweight, fast, and performed surprisingly well
SVM (RBF kernel)	   ~0.83	Strong performance, slightly below RF

**Each model was evaluated using:**
Confusion Matrix
Classification Report (Precision, Recall, F1-score)
Random Forest and Naïve Bayes showed the best overall performance.
