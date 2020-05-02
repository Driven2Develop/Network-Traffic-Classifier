import csv
import os
import pandas as pd
import numpy as np
import itertools as IT
import math
import logging
import sklearn
from sklearn import model_selection, svm, ensemble, tree

#Run the decision tree algorithm from the training sample and run the model on the test data
def getDecisionTreeLearningModel(train, test):
    treeModel = tree.DecisionTreeClassifier()
    treeModel = treeModel.fit(train, train["CLASSIFICATIONLABEL"])
    return pd.Series(treeModel.predict(test))

#repeat for Support Vector Machines (SVM)
def getSupportVectorMachinesModel(train, test):
    svmModel = svm.SVC()
    svmModel = svmModel.fit(train, train["CLASSIFICATIONLABEL"])
    return pd.Series(svmModel.predict(test))

#repeat for gradient boost tree
def getGradientBoostTreeModel(train, test):
    gradientModel = ensemble.GradientBoostingClassifier()
    gradientModel = gradientModel.fit(train, train["CLASSIFICATIONLABEL"])
    return pd.Series(gradientModel.predict(test))

#logging information
logLoc = "Docs\\Logs\\ModelRunner.log"
logging.basicConfig(filename = logLoc ,format='%(levelname)s : %(asctime)s : %(message)s', level=logging.DEBUG, datefmt= "%I:%M:%S")
logging.info("Log Successfully created")

#dataframe reference
dfref = "Dataset\\ProcessedDataset\\FeatureGenerator.csv"
learningColumns = pd.read_csv(dfref, nrows=1)
learningColumns = list(learningColumns.columns.values)

#remove unnecessary columns from the dataframe
learningColumns.remove("STARTTIME")
learningColumns.remove("ENDTIME")
learningColumns.remove("SRCADDRESS")
learningColumns.remove("DSTADDRESS")
learningColumns.remove("IPLABEL")

#import the dataframe excluding the unnecessary features. 
#divide the dataframe into training and testing with a 70-30 split respectively
df = pd.read_csv(dfref, usecols=learningColumns)
logging.info("dataset loaded into memory")
train, test = sklearn.model_selection.train_test_split(df, test_size=0.3)
logging.info("dataset split into 30% testing and 70% training")

#run the model
model = pd.DataFrame(test)
treelabel = getDecisionTreeLearningModel(train, test)
logging.info("Decision tree created from training set.")
svmlabel = getSupportVectorMachinesModel(train, test)
logging.info("SVM created from training set.")
gradientLabel = getGradientBoostTreeModel(train, test)
logging.info("Gradient Boost Tree created from training set.")

#append the models' classifiers to the original dataset
model["DECISIONTREEPREDICTEDLABEL"] = treelabel
logging.info("Decison tree predicted classifiers appended to testing set.")
model["SVMPREDICTEDLABEL"] = svmlabel
logging.info("SVM predicted classifiers appended to testing set.")
model["GRADIENTTREEPREDICTEDLABEL"] = gradientLabel
logging.info("Gradient Boost tree predicted classifiers appended to testing set.")

#save the model
model.to_csv("Dataset\\testing\\modelPredictedLabels.csv")
logging.info("Saving results of testing predicted labels.")