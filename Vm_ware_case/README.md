# Understanding Customer behavior at VMware

### Objective

VMware had various software, cloud and management products; it enabled enterprises to apply a software defined approach to business and IT.They have both digital and non- digital data for their customers and digital data for non-customers.The objective here is to use predictive analytics model to forecast the most appropriate digital actions for each customer. 

### Description

We have 707 features and 6 target actions. The effect variables are various digital actions - a multi-class response model. However the dataset is highly imbalanced since multi-class variables are highly sparse with almost 90% of data belonging to a single class label. Also the dataset has plenty of missing and unknown entries.

I used package VIM for missing values analysis and LASSO regression technique for variable selection. Since the Target class is highly imbalanced, I used SMOTE resampling method in R to do over and under-sampling of the class label and bring a balance to the dataset to some extent.

I used stacking ensemble technique with Randomforests and Logistic regression as base learners and again using Gradient boosting as Meta-learner to improve the model performance. I used Recall as the performace evaluator since there is imbalance in the target class variable 

### Technology

R




