# Spam-email-classification-RandomForest-GradientBoosting
This creates a classifier model that classifies the emails/messages in spam and ham category. The goal is to categorize the mails as spam and predict what future mails are going to be spam. I have used Bagging and Boosting Ensemble Technique for classification: Random Forest Classifier and Gradient Boosting Classifier; I have further tuned hyperparameters and evaluated the model with GridSearchCV.

## Machine Learning Piepline:

### 1. Read in  raw text
### 2. Clean Text and Tokenize
### 3. Feature Engineering
### 4. Fit Simple Model 
### 5. Tune hyperparameters and evaluate with GridSearchCV
### 6. Final Model Selection

## Data Pre-Processing:

Dataset includes text messages with label "spam" and "ham". The Question is to predict which future email/text should go into spam folder.
Data pre processing include tokenization , converting text into lower case, removing punctuation, stemming (PorterStemmer) and removing stopword. There is a trade off between Stemming and Lemmatization depending on the business problem.  

Data is then Vectorized i.e converted into vectors. I have done analysis on both : CountVectorizer and TF-IDF.

## Feature Engineering:

In this step, I will try to figure out the latent feature that might be useful towards solving my business problem. Looking into the problem and the dataset, I can think of two features that might have an impact on the prediction:

* Length of the email: A general hypothesis is that spam emails are generally lengthier. 
* Number of punctuations in the text email: A geneal Hypotesis is that spam emails will have more number of punctuations.

### Evaluate the importace of these newly created featues:

#### Plot the length of the text feature:

![image](https://user-images.githubusercontent.com/54689111/82852099-e70c8e00-9ecf-11ea-940d-3139f1017b2d.png)

This shows that most of the spam emails are lengthier, thus satisfying the general hypothesis.

#### Plot the number of punctuations feature:

![image](https://user-images.githubusercontent.com/54689111/82852223-52eef680-9ed0-11ea-9c28-74f706271ee6.png)

This again shows the spam emails have more punctuations.

Let's look into thieir distributions:

![image](https://user-images.githubusercontent.com/54689111/82852282-8df12a00-9ed0-11ea-9785-71d7a45acbab.png)

![image](https://user-images.githubusercontent.com/54689111/82852295-98132880-9ed0-11ea-9f01-cbc1a75c5df7.png)

I have added these two features with my existing dataframe.

## Fit & Tune Hyperparameters for Random Forest Classifier Model with GridSearch CV:

### Step 1:

Explore this model through Holdout Set. I kept the hyperparameters as :
* n_estimators: 50
* max_depth: 50
* n_jobs: -1
* test_size: 0.2

![image](https://user-images.githubusercontent.com/54689111/82852698-9a29b700-9ed1-11ea-9552-fbb2a54523a9.png)


As, we can see that Precision is low i.e a lot pf spam emails are going into non-spam folder, which is not good.

### Step 2:

Tune the Hyperparametes with GridSearch CV. I have kept the range of n_estimators to be [10, 150, 300] and max_depth to be [30, 60, 90, None]. Also, this will include cross-validation with k=5. I introduced two time patameters: one fo calculating the fit time and another to calculate the predict time. This process will produce precision, accuracy, recall, fit time, predict time for each combination of n_estimators and max_depth. I have sorted the results with mean_test_score in decreasing order.

![image](https://user-images.githubusercontent.com/54689111/82853428-c1818380-9ed3-11ea-88cb-0b84d83e06f1.png)

Depending on the business problem, I will go with the appropriate combination of hyperparameters.

## Fit & Tune Hyperparameters for Gradient Boosting Classifier model with GridSearch CV:

### Step 1:

Explore this model through Holdout Set. I kept the hyperparameters as :
* n_estimators: Default
* max_depth: Default
* Learning Rate: 0.1
* test_size: 0.2

### Step 2:

Tune the Hyperparametes with GridSearch CV. I have kept the range of n_estimators to be [100, 150] and max_depth to be [7, 11, 15]. Also, this will include cross-validation with k=5. I introduced two time patameters: one fo calculating the fit time and another to calculate the predict time. This process will produce precision, accuracy, recall, fit time, predict time for each combination of n_estimators and max_depth. I have sorted the results with mean_test_score in decreasing order.

![image](https://user-images.githubusercontent.com/54689111/82854201-ba5b7500-9ed5-11ea-831a-f5c3b5765007.png)

Depending on the business problem, I will go with the appropriate combination of hyperparameters.

## Model Selection:

Then, I trained the model with the appropriate hyperparameters and found the results as below:

### Ranom Forest Classifier:

![image](https://user-images.githubusercontent.com/54689111/82854281-f55da880-9ed5-11ea-9ed5-8e8e4e44255f.png)

### Gradient Boosting CLassifier:

![image](https://user-images.githubusercontent.com/54689111/82854306-0efef000-9ed6-11ea-867c-16be06930f58.png)

Now, depending on the Business Problem trade-off can be done between Predict time, Precision and Recall among the two models.

