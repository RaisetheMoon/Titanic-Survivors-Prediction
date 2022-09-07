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
<img src="graphs/data distribution - all.jpg">

<img src="graphs/confusion matrix.png">

## Modeling
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

- The engineering of the right features is absolutely key.
  The goal there is to create the right categories between survived and not survived. They do not have to be the same size or equally distributed. What helped best is to group together passengers with the same survival rates.

- I tried many, many different algorightms.
Many overfit the training data (up to 90%) but do not generate more accurate predictions with the test data. What worked better is to use the cross-validation on selected algotirhms. It is OK to select algorithms with various results as there is strenght in diversity.

- Lastly, the right ensembling was best achieved with a votingclassifier with soft voting parameter
