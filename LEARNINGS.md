## pandas stuff

### pandas melt: 
use pandas melt to transform a df from wide to long format
I had my df in wide format. so the columsn were: idx, human_code, gpt_code, qwen_code ...
But i needed seperate rows per code snippet. With melt I transform it to idx, code, source


### groupshufflesplit:
When you have a df that has multiple rows that belong to the same group for example coding task -> multiple rows with different solutions to that task. 
With groupshufflesplit you can make the train/test split with rows of the same group not ending up in test AND train. This avoids data leakage. 


## CountVectorizer:
Takes documents and makes vectors with the count of the tokens as dimensions. Simple example: "hello cat. Bye cat." would become (1, 2, 1)
Step 1 fit: finds all unique tokens and builds a vocabulary
Step 2 transform: counts how many times the tokens appear and creates a vector per document

## TfidfTransformer:
Takes count matrix (multiple vectors) and compares the frequency of each token with the num of appearances of that token in all the vectors.
TF: How many times does this token appear in a document?
IDF: In how many different documents does this token appear in?
Step 1 fit: calculates the inverse document frequency / weights. So (num of documents / num of document where token appears)
Step 2 transform: Looks at a single document and multiplies the tokens in the count vector with
the corresponding idf weight calculated in Step 1. Forms and normalises a vector.


## ROC Curve
Compares the test_y and the predictet probability. 
calculates the fpr and tpr for each threshold

## FixedThresholdClassifier
Wraps over a LogisticRegression Model and change the threshold value. Normally its fixed to 0.5 but with the ftc you can set it yourself. 


## Steps
1. Training: Model learns using the tf-idf
2. hyperparameter optimization using loss function
3. threshold tuning

## cross validation
technique to evaluate how well a model works. instead of realying on single train/test split it
repeatedly tests the model on different rounds / folds of train/test splits. 
Why: a single split could eg. produce an easy test split thus overestimate the models ability. 

cv=5 -> produces 5 cross validation folds

## Pipeline 
Sklearn Pipeline puts data-processing and a final model into a single object. With a Pipeline the entire process, from raw data to prediction, is executed, tuned, and can be reused with multiple datasets using a single command.
Every step except the last one has to be a transformer. Last one can be a classifier for example

