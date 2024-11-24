
# Project: Home Credit Default

Project is about me and my friend brilliant startup idea - provide risk evaluation as a service for retail banks. We were provided with various datasets from the banks and it will be used to analyse given data and create machine models to predict whether or not individual should get loan with his parameters.


## Data source

Given data source can be found at Kaggle with the link provided below:

  **Home credit default**  

https://storage.googleapis.com/341-home-credit-default/home-credit-default-risk.zip



## Summary
 
Whole project target was to create a reliable ML model that  
could predict with a given data if a client could or should  
receive a loan. These are the insights of project:  
- There are 9 datasets with various amount of rows and columns  
- data_train dataset shape is 307511, 122  
- data_train dataset had many things to clean, so various of methods were used: 
    - removed features that had more than 40% NaN values  
    - binary values were transformed to categorical to be more useful when dealing with tree based ML models  
    - NaN values were filled using self made .nan_imput function  
    - small details fixed with self made .cleaner function
- On EDA part data showed that only 8% of individuals have received credit/loan while 92% did not. This is strong indicator of class imbalance which will require extra steps and careful choices in ML part  
- Some of numeric data negative days were transformed to years for more clear view  
- For some analysis, null hypothesis were used:  
    - D'Agostino's K-squared test  
    - Two-Sample T-Test  
    - Levene's test  
- Many numeric data features were distributed with the skeweness. To make distribution towards normal, these features were transformed to logaritmic values  
- One of interesting discovery in categorical data is that the highest percent of received loans per occupation were low-skill laborers. Usually these jobs aren't very highly paid meaning eighter occupation isn't the main feature criteria for loan or the applications for loan were smaller ones  
- On feature engineering part few new columns were created from another dataset - bureau. New features: individual's active credits, credit sum debt, sum of prolonged credits  
- Degree of dependency was checked on target variable for categorical data using mutual info score function. None of features were highly depentant  
- Multicollinearity of numeric features using Variance Inflation Factor (VIF) was checked. Featerues like amt_credit,amt_income_total, amt_goods_price, amt_annuity showed very high VIF values. Some of thse were removed which helped to decrease VIF values significantly  
- For machine modeling part 5 models were selected: KNN, Logistic Regression, SVC, Random Forest, XGBoost  
- Columns were transformed and preprocessors for catergorical -  OneHotEncoder, numerical - StandardScaler were used  
- Data was cross-validated for more accurate results  
- Best performed model was RandomForestClassifier() which was used further with hyperparameters  
- Due to high class imbalance in target, RF class_weight hyperparameter was set to "balanced". This technique is similar to SMOTE which is used in other models  
- Despite using balanced parameters, model still, showed high f1-score to 0's and low (0.27) to 1's, meaning model is more accurate to predict 0's (individuals which would receive loan)  
- Model was deployed to GCP. Method to prepare deployment was used FastAPI + Dockerfile  
  
**What could have been done more:**  
- H0 hypothesis with different features could have been analyzed  
- Visualize summarized ML results to look for more insights  
- Try SHAP, ELI5, LIME to find which features impacts more highly