# Customer Behavior Analytics in VMware using R

VMware had various software, cloud and management products; it enabled enterprises to apply a software defined approach to business and IT.They have both digital and non- digital data for their customers and digital data for non-customers.

The objective here is to use predictive analytics model to forecast the most appropriate digital actions for each customer. The effect variables are various digital actions - a multi-class response model. However the dataset is highly imbalanced since multi-class variables are highly sparse.

We have 707 features and 6 target actions. The data needs initial pre-processing and variable reduction to build an efficient model.

I used package VIM for missing values analysis and LASSO regression for variable selection. I used stacking ensemble technique with Randomforests, Logistic regression and gradient boosting as base learners and again using Gradient boosting as Meta-learner.
