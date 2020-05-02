import csv
import os
import pandas as pd
import numpy as np

# the data input in this case is a large csv. 
# This program produces a log file with all the execution times, and a model that I can load afterwards.

def DataReader (datasetPath):

    #read the dataset
    df = pd.read_csv(datasetPath)
    # test by printing the number of rows
    print("Dataset contains " + str(df.STARTTIME.size) + " rows.")
    #send the dataset to the data processor 
    DataProcessor(df)

def DataProcessor (df):
    timeHeaders =  ["ENDTIME", "STARTTIME"] #headers that will be converted to timestamps
    countHeaders = ["PACKETINCOUNT", "PACKETINCOUNT", "PACKETOUTCOUNT", "BYTEINCOUNT"] #headers that will be converted to int32
    flowHeaders = ["TRANSPORTFLAGS", "FLOWS"] #Headers that will be converted to int8

    for column in df:
        #if the column lies in the time headers array then convert to timestamp.
        if column in timeHeaders:
            df[column] = pd.to_datetime(df[column])
            print (df[column])

        #if the column lies in the count headers array then convert to int32.
        if column in countHeaders:
            df[column] = df[column].astype(np.int32)
            print(df[column])

        #if the column lies in the flow headers array then convert to int8.   
        if column in flowHeaders:
            df[column] = df[column].astype(np.int8)
            print(df[column])

    df = df.sort_values("ENDTIME")  #sort the dataframe by endtime
    print (df)
    
    #add duration column as a engineered feature. and save the dataframe
    df = df.assign(DURATION = (df.ENDTIME-df.STARTTIME))
    df.to_csv("Dataset\\testing\\TrainModel.csv", encoding='utf-8', index=False)
    print(df)

#get Dataset path and headers
datasetPath = "Dataset\\TestDataSet.csv"
DataReader(datasetPath)