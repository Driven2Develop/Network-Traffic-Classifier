import csv
import os
import pandas as pd
import numpy as np
import logging
from shutil import copyfile

#the purpose of this program is to preprocess some of the data before generating the features. 
# With the exception of "DURATION", discretization and feature generation will be implemented in FeatureGenerator.py.

def DataPreProcessor (df):

    #only load in the columns of the copied dataset that need to be modified.
    headers = ["ENDTIME", "STARTTIME", "PACKETINCOUNT", "PACKETINCOUNT", "BYTEINCOUNT", "TCPFLAGS", "FLOWS"]

    #select the headers that will be converted to timestamps, int32, int8 respectively
    timeHeaders =  ["ENDTIME", "STARTTIME"]
    countHeaders = ["PACKETINCOUNT", "PACKETINCOUNT", "PACKETOUTCOUNT", "BYTEINCOUNT"]
    flowHeaders = ["TCPFLAGS", "FLOWS"]

    #Iterate through the columns, if they lie in the specified headers above convert them to a more suitable type
    for column in df:
        if column in timeHeaders:
            df[column] = pd.to_datetime(df[column])

        if column in countHeaders:
            df[column] = df[column].astype(np.int32)

        if column in flowHeaders:
            df[column] = df[column].astype(np.int8)

    logging.info("headers replaced")

    #add duration column as a engineered feature. Convert the duration to an integer then save the dataframe locally.
    df = df.assign(DURATION = (df.ENDTIME-df.STARTTIME))
    df["DURATION"]=df["DURATION"].astype('timedelta64[s]')
    df["DURATION"]=df["DURATION"].astype(int)

    logging.info("DURATION column added")
    return df
 
#logging information
logLoc = "Docs\\Logs\\TrainModel.log"
logging.basicConfig(filename = logLoc ,format='%(levelname)s : %(asctime)s : %(message)s', level=logging.DEBUG, datefmt= "%I:%M:%S")
logging.info("Log Successfully generated")

srcDatasetPath = "Dataset\\FullDataSet.csv"
dstDatasetPath = "Dataset\\ProcessedDataset\\Preprocessing.csv"
#copyfile(srcDatasetPath,dstDatasetPath)

#get the headers and append to a new dataframe
headers = pd.read_csv(srcDatasetPath, nrows=1).columns.values
featureHeaders =["DURATION"]
headers = np.concatenate((headers, featureHeaders), axis =0)

preprocDf = pd.DataFrame(columns = headers)
preprocDf.to_csv(dstDatasetPath, encoding='utf-8', index=False)

#Apply the preprocessing and append to the destination csv file
chunksize = 10000
for chunk in pd.read_csv(srcDatasetPath, chunksize=chunksize):
    logging.info("preprocessing chunk")
    df = DataPreProcessor(chunk)

    logging.info("appending chunk to destination csv file")
    df.to_csv(dstDatasetPath, encoding='utf-8', index=False, header=False, mode='a')