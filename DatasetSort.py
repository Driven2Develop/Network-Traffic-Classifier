import csv
import os
import pandas as pd
import numpy as np
import logging
from shutil import copyfile

srcDatasetPath = "Dataset\\ProcessedDataset\\Preprocessing.csv"
dstDatasetPath = "Dataset\\ProcessedDataset\\SortedDataset.csv"

df = pd.read_csv(srcDatasetPath)
df = df.sort_values(by=["ENDTIME"])
df.to_csv(dstDatasetPath, encoding='utf-8', index=False)