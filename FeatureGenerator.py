import csv
import os
import pandas as pd
import numpy as np
import itertools as IT
import math
import logging

#the purpose of this program is to generate the necessary features to apply the machine learning algorithms
#Discretization is also handled in this program to optimize time and memory.

#function to discretize (bin) the source and destination ports into one of 10 integer values.
def binPorts(value):
    maxPortCount = 64000
    binCount = 10
    ratio = math.ceil(value/maxPortCount*binCount)/binCount  #ceiling of the first decimal place
    return int(ratio*binCount)

#This helper method returns a dictionary of data that will be used to populate the connection based features last row.
#WindowIndex in this case refers to the last row of the dataframe.
def CreateTimeBasedFeatures (dfref, dfSize, windowIndex, endTime):
    
    #size of the rolling window based on time
    #will be increased to 10 minutes in full dataset
    startTime = pd.to_datetime(endTime) - pd.offsets.Minute(10)
    
    #get the rows that are within the time frame [startTime, endTime] based on the 'ENDTIME' column
    chunkSize = 10000 #can be increased in full dataset
    chunkIndex = 1
    df = pd.DataFrame(columns = headers)
    
    while chunkIndex<dfSize:
        chunks = pd.read_csv(dfref, skiprows=range(1,chunkIndex), nrows=chunkSize)
        
        #if the first end time in the chunks is greater than the end time of the time frame break out of loop
        if (chunks["ENDTIME"].iloc[0] > endTime):
            break
        #search for the rows where 'ENDTIME' lies within the time frame and append them to a temporary dataframe
        chunks['ENDTIME'] = pd.to_datetime(chunks['ENDTIME'])
        mask = (chunks['ENDTIME'] >= startTime) & (chunks['ENDTIME'] <= endTime)
        df = df.append(chunks.loc[mask])

        #increase the index to look through the next chunk
        chunkIndex = chunkIndex + chunkSize

    #get the rows that are equal to the last source address in the time based dataframe.
    targetSourceAddress = df["SRCADDRESS"].iloc[-1]
    index = df.index[df["SRCADDRESS"] == targetSourceAddress]
    srcdf = df.loc[index]

    #do the same for the last destination address
    targetDestinationAddress = df["DSTADDRESS"].iloc[-1]
    index = df.index[df["DSTADDRESS"] == targetDestinationAddress]
    dstdf = df.loc[index]

    #populate the last row of dataframe with the engineered features
    timeBasedValues = {
    "TIME_BASED_SRCADDRESS_TOTAL_OCCURENCES": len(df.index),
    "TIME_BASED_SRCADDRESS_OCCURENCES":len(srcdf.index),
    "TIME_BASED_SRCADDRESS_DISTINCT_DSTADDRESS":srcdf["DSTADDRESS"].nunique(),
    "TIME_BASED_SRCADDRESS_DISTINCT_DSTPORTS":srcdf["DSTPORT"].nunique(),
    "TIME_BASED_SRCADDRESS_DISTINCT_SRCPORTS":srcdf["SRCPORT"].nunique(),
    "TIME_BASED_SRCADDRESS_AVGPACKETIN":srcdf["PACKETINCOUNT"].mean(),
    "TIME_BASED_SRCADDRESS_AVGBYTEIN":srcdf["BYTEINCOUNT"].mean(),

    #repeat for the destination address
    "TIME_BASED_DSTADDRESS_TOTAL_OCCURENCES":len(df.index),
    "TIME_BASED_DSTADDRESS_OCCURENCES":len(dstdf.index),
    "TIME_BASED_DSTADDRESS_DISTINCT_SRCADDRESS":dstdf["SRCADDRESS"].nunique(),
    "TIME_BASED_DSTADDRESS_DISTINCT_DSTPORTS":dstdf["DSTPORT"].nunique(),
    "TIME_BASED_DSTADDRESS_DISTINCT_SRCPORTS":dstdf["SRCPORT"].nunique(),
    "TIME_BASED_DSTADDRESS_AVGPACKETIN":dstdf["PACKETINCOUNT"].mean(),
    "TIME_BASED_DSTADDRESS_AVGBYTEIN":dstdf["BYTEINCOUNT"].mean()

    }
    #return the last row of the dataframe  that will contain all the information about rolling window
    return timeBasedValues

#Function used to create connection based features based off a rolling window of size 1000.
def CreateConnectionBasedFeatures (dfref, dfSize, targetDfRef, headers):

    #rows 0-windowSize will remain blank for the connection based features.
    windowSize = 1000
    windowIndex = 1
    
    while windowIndex < dfSize:

        #fetch a subset of the dataframe of length windowsize
        df = pd.read_csv(dfref, skiprows=range(1,windowIndex), nrows=windowSize)
        saveRow = pd.DataFrame(df, columns = headers).iloc[-1]

        #save the source and destination IP address of the last row of the rolling window
        targetSourceAddress = df["SRCADDRESS"].iloc[-1]
        targetDestinationAddress = df["DSTADDRESS"].iloc[-1]

        #The rolling window is of size windowSize and looks at the previous rows from the index that match the source address of the target.
        #get the rows that are equal to the target source address
        srcIndex = df.index[df["SRCADDRESS"] == targetSourceAddress]
        srcdf = df.loc[srcIndex]
        
        #repeat for destination address
        dstIndex = df.index[df["DSTADDRESS"] == targetDestinationAddress]
        dstdf = df.loc[dstIndex]

        #repeat for when source and destination address are the same. 
        srcAndDstdf = df.loc[(df['SRCADDRESS'] == targetSourceAddress) & (df['DSTADDRESS'] == targetDestinationAddress)]

        #from the rolling window fill out the engineered features for the source address.
        saveRow["CONN_BASED_SRCADDRESS_OCCURENCES"] = len(srcdf.index)
        saveRow["CONN_BASED_SRCADDRESS_OCCURENCES"] = len(srcdf.index)
        saveRow["CONN_BASED_SRCADDRESS_DISTINCT_DSTPORTS"]= srcdf["DSTPORT"].nunique()
        saveRow["CONN_BASED_SRCADDRESS_DISTINCT_DSTADDRESS"] = srcdf["DSTADDRESS"].nunique()
        saveRow["CONN_BASED_SRCADDRESS_DISTINCT_SRCPORTS"] = srcdf["SRCPORT"].nunique()
        saveRow["CONN_BASED_SRCADDRESS_AVGPACKETIN"] = srcdf["PACKETINCOUNT"].mean()
        saveRow["CONN_BASED_SRCADDRESS_AVGBYTEIN"] = srcdf["BYTEINCOUNT"].mean()

        #repeat the features for the destination address.
        saveRow["CONN_BASED_DSTADDRESS_OCCURENCES"] = len(dstdf.index)
        saveRow["CONN_BASED_DSTADDRESS_DISTINCT_DSTPORTS"] = dstdf["DSTPORT"].nunique()
        saveRow["CONN_BASED_DSTADDRESS_DISTINCT_SRCADDRESS"] = dstdf["DSTADDRESS"].nunique()
        saveRow["CONN_BASED_DSTADDRESS_DISTINCT_SRCPORTS"] = dstdf["SRCPORT"].nunique()
        saveRow["CONN_BASED_DSTADDRESS_AVGPACKETIN"] = dstdf["PACKETINCOUNT"].mean()
        saveRow["CONN_BASED_DSTADDRESS_AVGBYTEIN"] = dstdf["BYTEINCOUNT"].mean()

        #repeat for the destination and source address. 
        saveRow["CONN_BASED_DSTANDSRC_OCCURENCES"] = len(srcAndDstdf.index)
        saveRow["CONN_BASED_DSTANDSRC_DISTINCT_DSTPORTS"] = srcAndDstdf["DSTPORT"].nunique()
        saveRow["CONN_BASED_DSTANDSRC_DISTINCT_SRCPORTS"] = srcAndDstdf["SRCPORT"].nunique()
        saveRow["CONN_BASED_DSTANDSRC_AVGPACKETIN"] = srcAndDstdf["PACKETINCOUNT"].mean()
        saveRow["CONN_BASED_DSTANDSRC_AVGBYTEIN"] = srcAndDstdf["BYTEINCOUNT"].mean()

        #use helper method to get the timebased features.
        #The time based features will begin at the last index of the dataframe for the connection based features.
        #the helper method will return a dictionary, 
        #so use the values from the dictionary to populate the time based features in the last row of the dataframe
        timeBasedRow = CreateTimeBasedFeatures(dfref, dfSize, windowIndex + windowSize, saveRow["ENDTIME"])
        saveRow["TIME_BASED_SRCADDRESS_TOTAL_OCCURENCES"] = timeBasedRow["TIME_BASED_SRCADDRESS_TOTAL_OCCURENCES"]
        saveRow["TIME_BASED_SRCADDRESS_OCCURENCES"] = timeBasedRow["TIME_BASED_SRCADDRESS_OCCURENCES"]
        saveRow["TIME_BASED_SRCADDRESS_DISTINCT_DSTADDRESS"] = timeBasedRow["TIME_BASED_SRCADDRESS_DISTINCT_DSTADDRESS"]
        saveRow["TIME_BASED_SRCADDRESS_DISTINCT_DSTPORTS"] = timeBasedRow["TIME_BASED_SRCADDRESS_DISTINCT_DSTPORTS"]
        saveRow["TIME_BASED_SRCADDRESS_DISTINCT_SRCPORTS"] = timeBasedRow["TIME_BASED_SRCADDRESS_DISTINCT_SRCPORTS"]
        saveRow["TIME_BASED_SRCADDRESS_AVGPACKETIN"] = timeBasedRow["TIME_BASED_SRCADDRESS_AVGPACKETIN"]
        saveRow["TIME_BASED_SRCADDRESS_AVGBYTEIN"] = timeBasedRow["TIME_BASED_SRCADDRESS_AVGBYTEIN"]

        saveRow["TIME_BASED_DSTADDRESS_TOTAL_OCCURENCES"] = timeBasedRow["TIME_BASED_DSTADDRESS_TOTAL_OCCURENCES"]
        saveRow["TIME_BASED_DSTADDRESS_OCCURENCES"] = timeBasedRow["TIME_BASED_DSTADDRESS_OCCURENCES"]
        saveRow["TIME_BASED_DSTADDRESS_DISTINCT_SRCADDRESS"] = timeBasedRow["TIME_BASED_DSTADDRESS_DISTINCT_SRCADDRESS"]
        saveRow["TIME_BASED_DSTADDRESS_DISTINCT_DSTPORTS"] = timeBasedRow["TIME_BASED_DSTADDRESS_DISTINCT_DSTPORTS"]
        saveRow["TIME_BASED_DSTADDRESS_DISTINCT_SRCPORTS"] = timeBasedRow["TIME_BASED_DSTADDRESS_DISTINCT_SRCPORTS"]
        saveRow["TIME_BASED_DSTADDRESS_AVGPACKETIN"] = timeBasedRow["TIME_BASED_DSTADDRESS_AVGPACKETIN"]
        saveRow["TIME_BASED_DSTADDRESS_AVGBYTEIN"] = timeBasedRow["TIME_BASED_DSTADDRESS_AVGBYTEIN"]

        #discretize the source and destination ports as well as 
        saveRow["SRCPORT"] = binPorts(saveRow["SRCPORT"])
        saveRow["DSTPORT"] = binPorts(saveRow["DSTPORT"])

        # Save the last row of the engineered features to a separate CSV file.
        # The method we are using to populate the values results in a saveRow being a series instead of a dataframe.
        # Increase index to next row. 
        windowIndex = windowIndex + 1
        saveRow.to_frame(name=None).transpose().to_csv(targetDfRef, mode = 'a', header = False, index=False)

#logging information
logLoc = "Docs\\Logs\\FeatureGenerator.log"
logging.basicConfig(filename = logLoc ,format='%(levelname)s : %(asctime)s : %(message)s', level=logging.DEBUG, datefmt= "%I:%M:%S")
logging.info("Log Successfully created")

#the file names of the preprocessed dataframe and the dataframe that will contain the new features.
dataframeReference = "Dataset\\ProcessedDataset\\SortedDataset.csv"
featureGeneratedDfRef = "Dataset\\ProcessedDataset\\FeatureGenerator.csv"

#get the headers of the dataframe and features then combine them.
columnNames = pd.read_csv(dataframeReference, nrows = 1)
columnNames = columnNames.columns.values
featureHeaders = [
"CONN_BASED_SRCADDRESS_OCCURENCES", "CONN_BASED_SRCADDRESS_DISTINCT_DSTPORTS", "CONN_BASED_SRCADDRESS_DISTINCT_DSTADDRESS", "CONN_BASED_SRCADDRESS_DISTINCT_SRCPORTS", "CONN_BASED_SRCADDRESS_AVGPACKETIN", "CONN_BASED_SRCADDRESS_AVGBYTEIN",
"CONN_BASED_DSTADDRESS_OCCURENCES", "CONN_BASED_DSTADDRESS_DISTINCT_DSTPORTS", "CONN_BASED_DSTADDRESS_DISTINCT_SRCADDRESS", "CONN_BASED_DSTADDRESS_DISTINCT_SRCPORTS", "CONN_BASED_DSTADDRESS_AVGPACKETIN", "CONN_BASED_DSTADDRESS_AVGBYTEIN",
"TIME_BASED_SRCADDRESS_TOTAL_OCCURENCES", "TIME_BASED_SRCADDRESS_OCCURENCES", "TIME_BASED_SRCADDRESS_DISTINCT_DSTADDRESS", "TIME_BASED_SRCADDRESS_DISTINCT_DSTPORTS", "TIME_BASED_SRCADDRESS_DISTINCT_SRCPORTS", "TIME_BASED_SRCADDRESS_AVGPACKETIN", "TIME_BASED_SRCADDRESS_AVGBYTEIN",
"TIME_BASED_DSTADDRESS_TOTAL_OCCURENCES", "TIME_BASED_DSTADDRESS_OCCURENCES", "TIME_BASED_DSTADDRESS_DISTINCT_SRCADDRESS", "TIME_BASED_DSTADDRESS_DISTINCT_DSTPORTS", "TIME_BASED_DSTADDRESS_DISTINCT_SRCPORTS", "TIME_BASED_DSTADDRESS_AVGPACKETIN", "TIME_BASED_DSTADDRESS_AVGBYTEIN",
"CONN_BASED_DSTANDSRC_OCCURENCES", "CONN_BASED_DSTANDSRC_DISTINCT_DSTPORTS", "CONN_BASED_DSTANDSRC_DISTINCT_SRCPORTS", "CONN_BASED_DSTANDSRC_AVGPACKETIN", "CONN_BASED_DSTANDSRC_AVGBYTEIN",
]
headers = np.concatenate ((columnNames, featureHeaders), axis =0)

#create new csv file to store the feature generated values and save the headers as the first row.
featureDf = pd.DataFrame(columns = headers)
featureDf.to_csv(featureGeneratedDfRef, encoding='utf-8', index=False)
logging.info("New CSV file created with headers")

#get number of rows from the original reference dataframe
# with open(dataframeReference) as f:
#     size = sum(1 for line in f)
#for simplicity
size = 14076197
logging.info("Number of rows obtained")

#create the connection based and time based features.
CreateConnectionBasedFeatures(dataframeReference, size, featureGeneratedDfRef, headers)
logging.info("all connection and time based features created")