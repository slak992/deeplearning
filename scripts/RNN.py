#Get the data from the dataset
import pandas as pd
import numpy as np
from huggingface_hub.keras_mixin import keras
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras import layers
import keras

google_stock_data_df = pd.read_csv('../dataset/Google_Stock_Price_Train.csv')

training_set = google_stock_data_df.iloc[:,1:2].values

#Feature scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_scaled= sc.fit_transform(training_set)
#training_scaled = minmax_scale(training_set, feature_range=(0,1))
#Creating training data set
x_tarin = []
y_train = []
for i in range(60,1258):
    x_tarin.append(training_scaled[i-60:i,0])
    y_train.append(training_scaled[i,0])
    print('done')

x_train, y_train = np.array(x_tarin), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

#Reshaping xtrain

# print('Hello')
#
# regressor = Sequential()
# regressor.add(layers.LSTM(units=50, return_sequences=True,input_shape=(x_train.shape[1],1)))
# regressor.add(layers.Dropout(0.2))
# regressor.add(layers.LSTM(units=50, return_sequences=True))
# regressor.add(layers.Dropout(0.2))
# regressor.add(layers.LSTM(units=50, return_sequences=True))
# regressor.add(layers.Dropout(0.2))
# regressor.add(layers.LSTM(units=50, return_sequences=False))
# regressor.add(layers.Dropout(0.2))
# regressor.add(layers.Dense(units=1))
#
# regressor.compile(optimizer='adam', loss='mean_squared_error')
# regressor.fit(x_train,y_train,epochs=100,batch_size=32)
#
#
# keras.saving.save_model(regressor, '../model/RNN.keras')

print("Done with the saving")

#Get the model

regressor_model = keras.saving.load_model('../model/RNN.keras')

#collecting and cleaning the test data

test_dataset_df = pd.read_csv('../dataset/Google_Stock_Price_Test.csv')
test_out = test_dataset_df.iloc[:,1:2].values
full_dataset_df = pd.concat((google_stock_data_df['Open'],test_dataset_df['Open']),axis=0)
input_test = full_dataset_df[len(full_dataset_df)-len(test_dataset_df)-60:].values
input_test= input_test.reshape(-1,1)
input_test =sc.transform(input_test)

x_test = []

for i in range(60,len(input_test)):
    x_test.append(input_test[i-60:i,0])
x_test = np.array(x_test)
inputs = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
preddiction = regressor_model.predict(inputs)
predict_value = sc.inverse_transform(preddiction)
print(predict_value)

#Visualization
from matplotlib import pyplot as plt
plt.plot(test_out,label='Google Stock Price',color='red')
plt.plot(predict_value,label='RNN Prediction',color='blue')
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.title("Stock Price Prediction")
plt.legend()
plt.show()