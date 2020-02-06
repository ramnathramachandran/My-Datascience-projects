# Online News popularity analysis for Mashable

### Objective

With the growth of the Internet in daily life, people are in a minute away to read the news or watch any entertainment or read articles of different categories.As the demand is increased even online platforms rivalry has increased. Due to this, every online platform is striving to publish the articles on their site which have great value and bring most shares. In this project, we have data produced by ‘Mashable’ where they collected data of around 39000 articles. The goals are:

• Predicting the number of shares an article can get it 

• Classifying the articles into different categories? 

• Which category of article should be published maximum for higher number of shares? 

• On What week-day What type of article should Mashable post more? 

• For different categories of articles what should be their min and max content length?

### Description

The dataset has 58 predictive attributes, 2 non predictive attributes and 1 target value which is the number of shares an article can get.
Some of the predictive attributes are Number of words in the content, Text subjectivity, Text polarity, Title subjectivity and Polarity, categorical attributes indicating whether an article was published on particular day, number of hyperlinks, number of images in the article etc.

After Exploratory Analysis it has been found that most of the predictors and even the target "Number of shares" distributions are skewed with almost 90% of the observations are on the left or right. 

As per our objective the maximum of shares of an article if the article has been shared the most of the time then the features of article has been so peculiar that it has reached maximum of shares. So those extreme values should not be considered as Outliers, instead should be treated as Anamolies.

From correlation Analysis it has been found that some of the features are highly correlated among thmeselves than with the target.Thus I removed those specific features and in addition performed Lasso Regression to perform futher features extraction.I used Multiple Linear regression and Neural networks model for predicting the number of shares.

For classifying the articles based on categories I transformed the existing binary variables indicating each article category into a single multi-class variable. I performed Random forests and Support vector classifier techniques to classify the articles into respective categories.

### Technology

Python

