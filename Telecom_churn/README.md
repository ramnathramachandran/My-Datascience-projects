# Customer churn analysis for cell2cell

### Objective

Customer acquisition and retention is a key concern for many industries, especially acute in the strongly competitive and quick growth telecommunications industry. Meanwhile, since the cost of retaining a good customer is much lower than acquiring a new one, it is very profit-effective to input valuable resource on the Retention Campaign.

Customers churn for various different reasons. Experience tells us that once the customer has made up their mind, retention becomes really hard. Therefore, managing churn in a proactive process is essential for the growth of the customer base and profit.

The primary goal of churn analysis is usually to create a list of contracts that are likely to be cancelled in the near future.

### Description

The Data Source I've chosen here is an open source data(cell2cell) by Teradata center for customer relationship management at Duke University.

Cell2Cell dataset is preprocessed and a balanced version provided for analyzing Process. consists of 51,047 instances and 58 attributes.

The features are categorized based on five dimensions:

1.Customer demography: Age, Tenure, Gender, Location, Zip code, etc.
2.Bill and payment: Monthly fee, Billing amount, Count of overdue payment, payment method, Billing type, etc.
3.Network/TV/Phone usage records: Network use frequency, network flow, usage time and period, internet average speed, In-net call duration,    Call type, etc.
4.Customer care/service: Service call number, service type, service duration, account change count
5.Competitors information: Offer detail under similar plan, etc

The dataset had many missing values and as a result of Exploratory Data Analysis few of the numerical variables had Outliers. But I didn't remove the outliers as the values are making sense to the data. I've done Mean imputation for the missing values in the dataset.

I had three goals while analysing the dataset:

1. Predict whether a customer would churn out of the company inorrder to focus on Customer Retention management - Classification problem.
2. Predict the Monthly Revenue loss the company would recur for losing a customer - Regression problem.
3. Perform Customer segmentation so that we can segment customers by profitability and target Customers only on those profitable segments - Clustering problem.

I used RandomForests and Logistic Regression for predicting Customer churn, Stochastic Gradient descent and Neural Networks for predicting Revenue loss and K-means clustering to cluster the Customers based profitability.

Here missing a churning customer would be much more costly than mislabeling a loyal customer as a churn risk. Thus Accuracy is not a valid metric to use for evaluation. So Recall heavy F- score would be better in evaluating the model.

### Technology



