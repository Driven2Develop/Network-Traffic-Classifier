import csv
import os
import pandas as pd
import numpy as np
import itertools as IT
import math
import logging
import sklearn
from sklearn import model_selection, svm, ensemble, tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.utils import multiclass

#logging information
logLoc = "Docs\\Logs\\AlgorithmResult.log"
logging.basicConfig(filename = logLoc ,format='%(levelname)s : %(asctime)s : %(message)s', level=logging.DEBUG, datefmt= "%I:%M:%S")
logging.info("Log Successfully created")

destLoc = "Results\\ClassifierResults.csv"
modelLoc = "Dataset\\ProcessedDataset\\modelPredictedLabels.csv"

#Load the dataframe, but just the classifier columns
classifierColumns = ["CLASSIFICATIONLABEL","DECISIONTREEPREDICTEDLABEL", "SVMPREDICTEDLABEL", "GRADIENTTREEPREDICTEDLABEL"]
df = pd.read_csv(modelLoc, usecols=classifierColumns)
logging.info("Classifiers loaded")

#there were some samples where the classifiers did not select a binary value, in this case make them 0
df = df.fillna(0)

#Initialize Dataframe
resultHeaders = ["Decision Tree Classifier", "SVM Classifier", "Gradient Boost Tree Classifier"]
resultDf = pd.DataFrame(columns = resultHeaders, index=["Accuracy", "Precision", "Recall"])
# resultDf["Metric"] = ["Accuracy", "Precision", "Recall"]
logging.info("Resultant dataframe initialized")

#calculate precision 
resultDf.set_value("Precision","Decision Tree Classifier", precision_score(df["CLASSIFICATIONLABEL"], df["DECISIONTREEPREDICTEDLABEL"])) 
resultDf.set_value("Precision","SVM Classifier", precision_score(df["CLASSIFICATIONLABEL"], df["SVMPREDICTEDLABEL"]))
resultDf.set_value("Precision","Gradient Boost Tree Classifier", precision_score(df["CLASSIFICATIONLABEL"], df["GRADIENTTREEPREDICTEDLABEL"]))
logging.info("Precision Calculated for all classifiers")

#calculate accuracy
resultDf.set_value("Accuracy", "Decision Tree Classifier", accuracy_score(df["CLASSIFICATIONLABEL"], df["DECISIONTREEPREDICTEDLABEL"]))
resultDf.set_value("Accuracy", "SVM Classifier", accuracy_score(df["CLASSIFICATIONLABEL"], df["SVMPREDICTEDLABEL"]))
resultDf.set_value("Accuracy","Gradient Boost Tree Classifier", accuracy_score(df["CLASSIFICATIONLABEL"], df["GRADIENTTREEPREDICTEDLABEL"]))
logging.info("Accuracy Calculated for all classifiers")

#calculate the recall
resultDf.set_value("Recall","Decision Tree Classifier", recall_score(df["CLASSIFICATIONLABEL"], df["DECISIONTREEPREDICTEDLABEL"]))
resultDf.set_value("Recall","SVM Classifier", recall_score(df["CLASSIFICATIONLABEL"], df["SVMPREDICTEDLABEL"]))
resultDf.set_value("Recall","Gradient Boost Tree Classifier", recall_score(df["CLASSIFICATIONLABEL"], df["GRADIENTTREEPREDICTEDLABEL"]))
logging.info("Recall Calculated for all classifiers")

#save the resulting dataframe
resultDf.to_csv(destLoc)