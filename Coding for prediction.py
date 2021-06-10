#Import all libraries required
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import requests
import openpyxl
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import LSTM

#Load data from local storage
path = r'*YOUR PATH TO THE FILE*' 
all_files = glob.glob(path + "/*.csv")     #Give the extension as per your datatype
li = []                    
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)                                                
frame = pd.concat(li, axis=0, ignore_index=True)
frame.set_index('Date and Time')
print(frame)

#Splitting data into train and test set
train_size = int(len(frame)*0.80)
train, test = frame[0:train_size], frame[train_size:len(frame)]
print(train)
print(test)

#Replacing NaN values with 0
train = train.replace(np.inf, np.nan)
test = test.replace(np.inf, np.nan)
train = train.fillna(0)
test = test.fillna(0)

#Normalizing data
cols_to_norm = ['Wind speed','Temperature', 'Relative Humidity','Air Pressure','Active Power']     #Give your column names
train[cols_to_norm] = train[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
test[cols_to_norm] = test[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

X_train = train.iloc[:,1:5]
y_train = train['Active Power'].values
X_test = test.iloc[:,1:5]
y_test = test['Active Power'].values

#Defining ANN parameters
nn_model = Sequential()
nn_model.add(Dense(8, input_dim=4, activation = 'relu'))
nn_model.add(Dense(2, activation = 'relu'))
nn_model.add(Dense(1))
nn_model.summary()

#Training ANN model
nn_model.compile(loss='mean_squared_error', optimizer='adam')
early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
history = nn_model.fit(X_train, y_train, epochs=10, batch_size=10, verbose=1, callbacks=[early_stop], shuffle=False)

#Predicting and testing the accuracy using R2 score
y_pred_test_nn = nn_model.predict(X_test)
y_train_pred_nn = nn_model.predict(X_train)
print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_nn)))
print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_nn)))

#Plotting predicted and actual values of the test set
plt.figure(figsize=(10, 6))
plt.plot(y_test, label='True')
plt.plot(y_pred_test_nn, label='NN')
plt.title("NN's Prediction")
plt.xlabel('Observation')
plt.ylabel('Active Power')
plt.legend()
plt.show()

#Calling API from weather website for weather forecast values
weather_key = '*YOUR KEY*'
url = '*YOUR API URL*'
params = {'APPID':weather_key, 'q':'Magdeburg', 'units':'imperial'}     #Give your location and the desired units
response = requests.get(url,params=params)
data = response.json()

#Extracting and saving the data into a dataframe
test_weather = pd.DataFrame(dtype = float)
header = ["Date and Time","Wind speed","Temperature","Relative Humidity","Air Pressure"]     #Give your column names
temperature = []
pressure = []
humidity = []
wind_speed = []
date = []
for i in range(len(data.get('list'))):
    #print(i)
    temperature.append(data.get('list')[i].get('main').get('temp'))
    pressure.append(data.get('list')[i].get('main').get('pressure'))
    humidity.append(data.get('list')[i].get('main').get('humidity'))
    wind_speed.append(data.get('list')[i].get('wind').get('speed'))
    date.append(data.get('list')[i].get('dt_txt'))
    
test_weather[header[0]] = date
test_weather[header[1]] = wind_speed
test_weather[header[2]] = temperature
test_weather[header[3]] = humidity
test_weather[header[4]] = pressure  
print(test_weather)

test_weather.set_index('Date and Time')
X_val = test_weather.iloc[:,1:5]
print(X_val)

#Making predictions with the weather forecast data and plotting results
nn_y_pred_val = nn_model.predict(X_val)
test_weather['Date and Time'] = pd.to_datetime(test_weather['Date and Time'])
plt.figure(figsize=(10, 6))
plt.plot(test_weather['Date and Time'], nn_y_pred_val, label='NN')
plt.title("NN's Prediction")
plt.xlabel('Date')
plt.ylabel('Active Power')
plt.legend()
plt.show()















