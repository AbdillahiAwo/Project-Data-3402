### **Loan Approval Project:**
This repository contains an attempt to apply classification algorithms to the Loan Approval Prediction dataset from the Kaggle Tabular Playground Challenge (Loan Approval Prediction | Kaggle).

### **Overview:**
- Goal: Predict loan approval using features like age, income, and credit history with logistic regression.
- Results: Achieved 90% accuracy and 67.04% precision after preprocessing and data splitting.

### **Summary of work done:**
Data
Type: Tabular data
- Dataset: 50,827 loan applications with 27 features (numerical & one-hot encoded categorical) and a binary target, "loan_status."
- Split: 70% training, 15% validation, 15% test to predict loan approval status.
  
### **Preprocessing / Clean up:**
- Preprocessing: Handled missing values, normalized numerics, and one-hot encoded categorical features (e.g., homeownership, loan intent, loan grade).
- Target/Columns: Converted "loan_status" to binary (0/1) and dropped redundant columns like "id."

### **Data Visualization:**
- Applicant Trends: Most applicants were aged 20-40, with debt consolidation as the top loan intent.
- Feature Insights: Larger loans had higher interest rates; income showed high variability with notable outliers.
![download](https://github.com/user-attachments/assets/2a8c4458-4c94-4858-9f1f-cf6c36a0a2cd)
![download-1](https://github.com/user-attachments/assets/442b4f4b-ed41-4abe-8131-91d0632373a6)


### **Problem Formulation:**
- Goal & Approach: Predict loan approval using features like age, income, and loan amount with models like Logistic Regression, Random Forest, and XGBoost.
- Model Choice: Logistic Regression was chosen for its simplicity and strong performance after tuning hyperparameters.

### **Training:**
- Trained models using Python, scikit-learn, and XGBoost on Google Colab with a MacBook M1 processor.
- Fitted Logistic Regression on processed data and validated performance on a separate validation set.

### **Performance Comparison:**
- Key metrics: Accuracy 90.4%, Precision 67.04%, Recall 32.02%, F1-score 43.34%, ROC AUC 84.54%.
- ROC curve shows good performance with high AUC, but recall can be improved.

### **Conclusions:**
- Logistic regression performed well with good accuracy and AUC, but had low recall for predicting approved loans.
- Future improvements could focus on enhancing recall with techniques like resampling or more complex models like random forests or XGBoost.

### **Future Work:**
- Plan to experiment with Random Forests, XGBoost, hyperparameter tuning, SMOTE, and feature engineering to improve recall for loan approvals.
- Future studies could explore deep learning, ensemble methods, and the impact of additional features like credit score or economic factors.

### **How to reproduce results:**
- Install libraries (pandas, numpy, scikit-learn, xgboost), preprocess data, and train a model (e.g., Logistic Regression, Random Forest).
- Evaluate the model on the validation set and test data, using Google Colab for cloud-based computation.

### **Overview of files in repository:**
- Repository includes Loan Approval Project.ipynb (for preprocessing, model training, evaluation, and submission), train.csv, and test.csv.
- All tasks are performed within the main notebook.

### **Software Setup:**
- Required packages: pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost, imbalanced-learn.
- Install with: pip install pandas numpy scikit-learn matplotlib seaborn xgboost imbalanced-learn (Google Colab may have them pre-installed).

### **Data:**
- Download the data (train.csv and test.csv) from the Kaggle Loan Approval Prediction Challenge page.
- Preprocess the data by handling missing values, encoding categorical variables, and separating features and target variable for model training.

### **Training:**
- Load and preprocess the training data, then split it into training and validation sets.
- Train the model (e.g., Logistic Regression), evaluate with metrics (accuracy, precision, recall), and optionally tune hyperparameters.

### **Performance Evaluation:**
- Evaluate the model using predictions on the validation set and metrics like accuracy, precision, recall, F1-score, and ROC AUC.
- Generate a confusion matrix and classification report to visualize and assess performance.

### **Citations:**
https://www.kaggle.com/competitions/playground-series-s4e10/data 
