#Loading libraries required for preapring data

import numpy as np
import pandas as pd
from numpy import split
from numpy import array
import math
import glob

#Load files from the local storage

path = r'*Put your path to the file*'
all_files = glob.glob(path + "/*.csv")     #Here '.csv' is used as the data received were in 'csv' format.
li = []
colnames=['Time','Total Active Power']     #Define your column names for the laoding data(optional)

for filename in all_files:
    df = pd.read_csv(filename, index_col=None,error_bad_lines=False,sep = ",",decimal='.', header=0)
    li.append(df)

dataframe= pd.concat(li, axis=0, ignore_index=True)

print(dataframe)       #To view the loaded dataframe

#Fixing one column as index and converting the dataframe to numpy array to perform resampling of data to the required duration between measurements.

dataframe["Time"] = pd.to_datetime(dataframe["Time"], errors="coerce")      #Here 'time' column is taken as the index of the dataframe.
dataframe = dataframe.set_index(['Time'])
file =dataframe.apply(pd.to_numeric, errors='coerce')
file = file.fillna(0)         #replacing NaN values in the dataframe with zero(optional)
file[file < 0] = 0            #replacing negative values with zero(this is subjective and optional)
file_resampled = file.resample('15min').mean()         #resampling the data to the required duration
final_file = file_resampled.interpolate(method='linear', limit_direction='forward', axis=0)      #linear interpolation is performed to fill the missing or empty values of the data   
print(final_file)     #to see the final version of data

final_file = final_file.reset_index()       #this is optional
final_file.to_csv("*Your filename*")        #saving file to local storage

#The cleaned and resampled saved file can then be used for applying prediction algorithms.

