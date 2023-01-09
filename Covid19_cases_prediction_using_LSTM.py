#%%
# Import packages

import pandas as pd
import os
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, classification_report, ConfusionMatrixDisplay, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

#%%
# Data loading

train_csv_path = os.path.join(os.getcwd(), 'Dataset', 'cases_malaysia_train.csv')
train_df = pd.read_csv(train_csv_path)

# %%
# Data inspection

print(train_df.info())
print(train_df.head())
print(train_df.describe().T)

# %%
# Data cleaning
# Changing the data type of cases_new column from object to float
train_df['cases_new'] = pd.to_numeric(train_df['cases_new'], errors='coerce')
print(train_df.info())
print(train_df.isna().sum())

#%%
# Dealing with NaN values in cases_new column by interpolation
train_df['cases_new'] = train_df['cases_new'].interpolate(method='polynomial', order=2)
print(train_df.isna().sum())
print(train_df.info())

#%%
plt.figure(figsize=(10,10))
plt.plot(train_df['cases_new'].values)
plt.ylabel('Number of cases')
plt.xlabel('Days')
plt.show()

# %%
# Feature selection

new_cases = train_df['cases_new'].values

# %%
# Normalize the range of the data before training 
mms = MinMaxScaler()

new_cases = mms.fit_transform(new_cases[::,None])

# %%
x = []
y = []
window_size = 30

for i in range(window_size, len(new_cases)):
    x.append(new_cases[i-window_size:i])
    y.append(new_cases[i])

# %%

x = np.array(x)
y = np.array(y)

# %%
# Split the dataset into train and test dataset

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=64)

#%%
# Model development using LSTM
nIn = x_train.shape[1:]
model = Sequential()

model.add(LSTM(64, return_sequences=True, input_shape=(nIn)))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dropout(0.1))
model.add(Dense(1, activation='relu'))

model.summary()
#%%
keras.utils.plot_model(model, show_shapes=True)

#%%
# Model compilation
model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mape'])

# %%
# tensorboard callback

logs_path = os.path.join(os.getcwd(), 'new_covid_cases_logs', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
es = EarlyStopping(monitor='val_loss', patience=50, verbose=1, restore_best_weights=True)
tb = TensorBoard(log_dir=logs_path)

# %%
# Model training
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=300, batch_size=1280, callbacks=[tb,es])

# %%
# Model analysis
# Data loading of test dataset
test_csv_path = os.path.join(os.getcwd(), 'Dataset', 'cases_malaysia_test.csv')
test_df = pd.read_csv(test_csv_path)

#%%
# Data inspection of test dataset
print(test_df.info())
print(test_df.isna().sum())

#%%
# Data cleaning by interpolation of test dataset to remove NaN values
test_df['cases_new'] = test_df['cases_new'].interpolate(method='polynomial', order=2)

print(test_df.info())
print(test_df.isna().sum())

#%%
plt.figure(figsize=(10,10))
plt.plot(test_df['cases_new'].values)
plt.ylabel('Number of cases')
plt.xlabel('Days')
plt.show()

#%%
# Concatenate the train data and the test data

concatenated = pd.concat((train_df['cases_new'], test_df['cases_new']))
concatenated = concatenated[len(train_df['cases_new']) - window_size :]

# min max transformation of the concatenated data
concatenated = mms.transform(concatenated[::,None])

plt.figure()
plt.plot(concatenated)
plt.show()

# %%
x_test_2 = []
y_test_2 = []

for i in range(window_size, len(concatenated)):
    x_test_2.append(concatenated[i-window_size:i])
    y_test_2.append(concatenated[i])

x_test_2 = np.array(x_test_2)
y_test_2 = np.array(y_test_2)

# %%
# Model prediction on test dataset
predicted_new_cases = model.predict(x_test_2)

#%%
plt.figure()
plt.plot(predicted_new_cases, color='red')
plt.plot(y_test_2, color='blue')
plt.legend(['Predicted', 'Actual'])
plt.show()

# %%
print(mean_absolute_percentage_error(y_test_2, predicted_new_cases))
print(mean_squared_error(y_test_2, predicted_new_cases))

# %%
# Model Analysis

y_pred = np.argmax(predicted_new_cases, axis=1)
y_test = np.argmax(y_test_2, axis=1)

#%%
# Classification report and confusion matrix
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm)

disp = ConfusionMatrixDisplay(cm)
disp.plot()

# %%
# Model saving

model.save('covid cases prediction model.h5')

#to save one hot encoder model
with open('covid cases prediction mms.pkl', 'wb') as f:
    pickle.dump(mms, f)
# %%
