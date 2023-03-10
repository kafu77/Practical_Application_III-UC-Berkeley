# Practical_Application_III-UC-Berkeley

# Overview:
In this practical application,The goal is to compare the performance of the classifiers, namely K Nearest Neighbor, Logistic Regression, Decision Trees,and Support Vector Machines. I utilized a dataset related to marketing bank products over the telephone. The dataset comes from the UCI Machine Learning repository https://archive.ics.uci.edu/ml/datasets/Bank%2BMarketing. It is one of the common and powerful dataset repository for machine learing. 

Data Set Information:
The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed.

Attribute Information:
Input variables:

##bank client data:

1 - age (numeric)
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
5 - default: has credit in default? (categorical: 'no','yes','unknown')
6 - housing: has housing loan? (categorical: 'no','yes','unknown')
7 - loan: has personal loan? (categorical: 'no','yes','unknown')

##related with the last contact of the current campaign:

8 - contact: contact communication type (categorical: 'cellular','telephone')
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

##other attributes:

12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
14 - previous: number of contacts performed before this campaign and for this client (numeric)
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')

##social and economic context attributes

16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
17 - cons.price.idx: consumer price index - monthly indicator (numeric)
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
20 - nr.employed: number of employees - quarterly indicator (numeric)

# The Task:
This dataset is related to direct marketing campaigns via phone calls from a Portuguese banking institution. The dataset was obtained from the UCI Machine Learning Repository.

There are four datasets:
1) bank-additional-full.csv with all examples (41188) and 20 inputs, ordered by date (from May 2008 to November 2010), very close to the data analyzed in [Moro et al., 2014]
2) bank-additional.csv with 10% of the examples (4119), randomly selected from 1), and 20 inputs.
3) bank-full.csv with all examples and 17 inputs, ordered by date (older version of this dataset with less inputs).
4) bank.csv with 10% of the examples and 17 inputs, randomly selected from 3 (older version of this dataset with less inputs).
The smallest datasets are provided to test more computationally demanding machine learning algorithms (e.g., SVM).
The classification goal is to predict if the client will subscribe (yes/no) a term deposit (variable y).

**Exploratory data anaylysis and data visualization**:

The dataset was explored through data visualization using various libraries such as Matplotlib, Seaborn, and Plotly. The distribution of the data was found to be skewed to the right. However, after applying a logarithmic transformation, the distribution became almost symmetrical. Seaborn's Kernel Density Estimate (KDE) plot was used to estimate the probability density function of the continuous age dataset. The relationships between different features were also analyzed using Seaborn's pair plot, joint plot, bar plot, box plot, and heatmap plot. It was observed from the heatmap plot that 'euribor3m' and 'emp.var.rate' have a strong positive relationship, while 'previous' and 'emp.var.rate' have a strong negative relationship. A box plot was used to identify that retired individuals have the highest average age who subscribed to the term deposit.


**Engineering features**:

The text describes the data transformation and preparation process for machine learning. The data is encoded using tools from scikit learn, such as OneHotEncoder and OrdinalEncoder. A pipeline is prepared for each classifier, including transformer and model. The data is split into X and y data set, with X containing all features and y containing the target. The data is then split into train and test data sets using a tool from scikit learn preprocessing.

**Baseline Model**:

The baseline model is used as a comparison for the actual model and can be either a Simple Baseline Model or Machine Learning Baseline Model. A dummy classifier is used to generate the baseline results, and different metrics are generated based on different dummy classifier strategies. Since the class of the data set is imbalanced, a stratified strategy is suitable for this scenario. The accuracy of the dummy classifier based on the stratified strategy is 0.805778101480942, which serves as a benchmark for the complex models.

**Simple Model**:

The work involved training a Logistic Regression model with default parameters, followed by K Nearest Neighbor, Decision Tree, and Support Vector Machine classifiers with their default parameters. The models were evaluated based on their accuracy and speed of training. The results showed that all models had similar accuracy, but the Support Vector Machine took the longest time to train. The Decision Tree model appeared to be overfitting as it had a high accuracy on the train dataset but a lower accuracy on the test dataset. The next step would be to tune the hyperparameters of the models for better performance.

**Improving the Model**:

After tuning the hyperparameters, the performance of each model has been evaluated using the accuracy score, precision, recall, and F1 score. The Logistic Regression model achieved an accuracy of 0.898208, which is slightly better than the default model. The K Nearest Neighbor model achieved an accuracy of 0.897174, which is similar to the default model. The Decision Tree model achieved an accuracy of 0.886798, which is slightly worse than the default model. The Support Vector Machine model achieved an accuracy of 0.899409, which is the same as the default model.

In terms of precision and recall, the Support Vector Machine model performed the best, followed by the Logistic Regression model, K Nearest Neighbor model, and Decision Tree model. The F1 score also showed that the Support Vector Machine model performed the best, followed by the Logistic Regression model, K Nearest Neighbor model, and Decision Tree model.

Overall, the Support Vector Machine model with the radial basis function kernel, gamma of 0.1, and C value of 1.0, performed the best among all the models.

**Improved Results**:

Overall, the Logistic Regression and Decision Tree models performed well with good accuracy scores on both the train and test data sets. The KNN and SVM models also had good accuracy scores but took longer to train. The Decision Tree model had a slight overfitting problem, which was improved after tuning the hyperparameters. The Logistic Regression model did not have a significant improvement after tuning hyperparameters, which suggests that the default parameters were suitable for this dataset.

**Findings**:

The Decision tree model was overfitting on the default parameters but improved after hyperparameter tuning. The Logistic Regression model performed well in terms of accuracy and training time, while KNN and SVM models also performed well but took longer to learn when a grid search was applied. Overall, it seems like hyperparameter tuning can improve the performance of the models, and different models have different strengths and weaknesses.

**Recommendation**:

The Logistic Regression model is the best model on both accuracy and time.






