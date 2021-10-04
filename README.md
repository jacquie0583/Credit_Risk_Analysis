# Credit Risk Analysis
## Overview
In 2019, more than 19 million Americans had at least one unsecured personal loan. That's a record-breaking number! Personal lending is growing faster than credit card, auto, mortgage, and even student debt. With such incredible growth, FinTech firms are storming ahead of traditional loan processes. By using the latest machine learning techniques, these FinTech firms can continuously analyze large amounts of data and predict trends to optimize lending.
In this project we have utilized Python to build and evaluate several machine learning models to predict credit risk. These skills allow the prediction of credit risk with machine learning algorithms which can help banks and financial institutions predict anomalies, reduce risk cases, monitor portfolios, and provide recommendations on what to do in cases of fraud.  Skills acquired:

  •	Explain how a machine learning algorithm is used in data analytics.
  
  •	Create training and test groups from a given data set.
  
  •	Implement the logistic regression, decision tree, random forest, and support vector machine algorithms.
  
  •	Interpret the results of the logistic regression, decision tree, random forest, and support vector machine algorithms.

  •	Compare the advantages and disadvantages of each supervised learning algorithm.
  
  •	Determine which supervised learning algorithm is best used for a given data set or scenario.
  
  •	Use ensemble and resampling techniques to improve model performance.
  
##  Analysis Reports Used to Predict Credit Risk

1.	Oversampling Models:  naive random oversampling algorithm and the SMOTE algorithm
2.	Undersample Model: Cluster Centroids algorithm
3.	Resample Model:  used the SMOTEENN algorithm SMOTEENN Algorithm
4.	Classification Report assessed the performance of two ensemble algorithms; training a Balanced Random Forest Classifier and an Easy Ensemble AdaBoost classifier; each algorithm Ensemble Classifiers Generate used the imbalanced_classification_report from imbalanced-learn.

## Resources:
  	Data Source: Module-17-Challenge-Resources.zip and LoanStats_2019Q1.csv
  
  	Data Tools: credit_risk_resampling_starter_code.ipynb and          credit_risk_ensemble_starter_code.ipynb.
  
  	Software: Python 3.9, Visual Studio Code 1.50.0, Anaconda 4.8.5, Jupyter Notebook 6.1.4 and Pandas
  
  	NumPy, version 1.11 or later
  
  	SciPy, version 0.17 or later
  
  	Scikit-learn, version 0.21 or later
  
  	imbalanced-learn package in our mlenv environment.


##  Results:  Resampling Models to Predict Credit Risk

### Oversampling

<p align="center">
   <img width="400" height="200" src="https://github.com/jacquie0583/Credit_Risk_Analysis/blob/main/Image%201.jpg">
</p
    
 
![Image 2](https://github.com/jacquie0583/Credit_Risk_Analysis/blob/main/image%202.jpg)  


  
### Undersampling
  
Also, testing an undersampling algorithms to determine which algorithm results in the best performance compared to the oversampling algorithms above. The undersampling of the data done by the Cluster Centroids algorithm. 
  
![Image 3](https://github.com/jacquie0583/Credit_Risk_Analysis/blob/main/Image%203.jpg)

  
### Over/Under Sampling: SMOTEENN
  
Another test combined over- and under-sampling algorithm to determine if the algorithm results in the best performance compared to the other sampling algorithms above. 
  
  
![Image 4](https://github.com/jacquie0583/Credit_Risk_Analysis/blob/main/Image%204.jpg)  
  
 
## Ensemble Classifiers to Predict Credit Risk 
  
We compared two ensemble algorithms to determine which algorithm results in the best performance. You will train a Balanced Random Forest Classifier and an Easy Ensemble AdaBoost classifier. 

  
<p align="center">
   <img width="400" height="200" src="https://github.com/jacquie0583/Credit_Risk_Analysis/blob/main/image%205.jpg">
</p  

  
![Image 6](https://github.com/jacquie0583/Credit_Risk_Analysis/blob/main/image%206.jpg)  


# Analysis
  
Based on the accuracy score, the Ensemble Classifiers proved to be the most precise. EasyEnsembleClassifierp provides a highest Score for all Risk loans. The precision is low or none for all the models. In general, above the 90% of the current analysis, utlizing EasyEnsembleClassifier will perform a High-Risk loan precision as a great value for the overall analysis.
The ML models process of fitting, reshaping, and training the same data is carried out significantly different. The evaluating parameters followed by a description of their origin follow: 

ACCURACY SCORE reports a percentage of precision of the predictions compared to the actual results. However, it is not enough just to see that results, especially with unbalanced data.
Equation: accuracy score = number of correct prediction/total number of predictions.

PRECISION is the measure of how reliable a positive classification is. A low precision is indicative of a large number of false positives. Equation: Precision = TP/(TP + FP)

RECALL is the ability of the classifier to find all the positive samples. A low recall is indicative of a large number of false negatives Equation: Recall = TP/(TP+FN)

FI SCORE is weighted average of the true positive rate (recall) and precision, where the best score is 1.0. Equation: F1 score = 2(Precision x Sensititivity)/(Precision + Sensitivity) The F1 Score equation is: 2*((precisionrecall)/(precision+recall)). It is also called the F Score or the F Measure. Put another way, the F1 score conveys the balance between the precision and the recall. The F1 for the All No Recurrence model is 2((0*0)/0+0) or 0.

