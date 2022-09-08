# Titanic-Survivors-Prediction
I created this Python notebook to record my first complete machine learning project - Titanic, the classic competition topic on Kaggle. Even it is already well-known, the subject still interested me. So I dived into it, studied all the new terms, new functions and new approaches, and captured the explanations in this notebook.

## Abstract
<img src="https://static.timesofisrael.com/atlantajewishtimes/uploads/2022/03/DT6RD9.jpg" width="600">

> The RMS Titanic, a luxury steamship, sank in the early hours of April 15, 1912, off the coast of Newfoundland in the North Atlantic after sideswiping an iceberg during its maiden voyage. Of the 2,240 passengers and crew on board, more than 1,500 lost their lives in the disaster.

The movie "Titanic" is also one of my favorite movies, but before I became a data analyst/scientist, I never thought about below questions - 

*"What sort of people are most likely survive than others? Can we use machine learning to predict the survival rate and utilize it to other similar events?"*

In a word, the goal of this project is to build a machine learning model to predict who survived after the tragedy by using their personal data (ie: name, age, gender, fare, class, etc).

## Approach
In this project, I will use Python with **scikit-learn** to build a machine learning model (Random Forest Classifier), which use continuous and categorical data from [Kaggle (Titanic - Machine Learning from Disaster)](https://www.kaggle.com/competitions/titanic) to predict which passengers survived the Titanic shipwreck

**Random Forest Model** is an exceptionally useful machine learning method based on bagging different decision trees that split on a subset of features on each split.
The reasons I selected Random Forest for this project are:

1. It handles <ins>binary, categorical, and numerical features</ins>. No need to rescale, there is very little pre-processing needs to be done;
2. <ins>Computation process is parallelizable</ins>, faster than boosted models which are sequential and would take longer to compute;
3. It is great with <ins>high dimensional data</ins>;
4. It is <ins>robust to outliers and non-linear data</ins>;
5. <ins>More accurate</ins> with balancing bias and variance by averaging all the decision trees.


## Exploratory Data Analysis

### Data distribution
<img src="graphs/data distribution-all.png">


### Feature against survival rate
<img src="graphs/features against survival rate.png">

### Correlation of raw data
<img src="graphs/correlation of raw data.png" width="500">
It's easy to tell that survival rate has a strong relationship with **Sex**, and weak relationship with **Pclass**, **Fare**, and **Cabin**, which three variables also have strong relationship with each other.


## Modeling

### Confusion Matrix
>Confusion matrix can explain a binary classification problem. It gives us the overall accuracy of the model by calculating the fraction of the total samples that were correctly classified by the classifier.  
<img src="graphs/confusion matrix.png">

- Accuracy: Overall, how often is the classifier correct?  
(TP+TN)/total = (528+285)/891 = 0.91

- Misclassification Rate: Overall, how often is it wrong? Equivalent to 1 minus Accuracy, also known as "Error Rate"  
(FP+FN)/total = (21+57)/891 = 0.09

- True Positive Rate: When it's actually survived, how often does it predict yes? Also known as "Sensitivity" or "Recall"  
TP/actual yes = 285/(57+285) = 0.83

- False Positive Rate: When it's actually not survived, how often does it predict survived?  
FP/actual no = 21/(21+528) = 0.04

- True Negative Rate: When it's actually not survived, how often does it predict not survived? Equivalent to 1 minus False Positive Rate, also known as "Specificity"  
TN/actual no = 528/(528+57) = 0.90


### Feature Importance
Knowing feature importance can help us

1. with **variable selection** - we can remove x variables that are not that significant which will shorten training time;
2. get a better understanding of the **model’s logic** with which we can improve the model by focusing only on the important variables;
3. sacrifice some accuracy for the sake of **interpretability** in some business cases

<img src="graphs/feature importance.png">
It seems that the top 5 most important features are:

- Title (Mr.)
- Fare
- Sex (Female)
- Sex (Male)
- Name Length

## Result
This model leading to a 79% accuracy in the end 

#### Some key take away from my personal experiments and what-if analysis:

- The engineering of the right features is absolutely key;
- Parameter tunning is also important.
