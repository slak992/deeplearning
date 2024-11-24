#Get the data from the dataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale

google_stock_data_df = pd.read_csv('../dataset/Google_Stock_Price_Train.csv')
training_set = google_stock_data_df.iloc[:,1:2].values

#Feature scaling
training_scaled = minmax_scale(training_set, feature_range=(0,1))
#Creating training data set
x_tarin = []
y_train = []
for i in range(60,1258):
    x_tarin.append(training_scaled[i-60:60,0])
    y_train.append(training_scaled[i,0])
    print('done')

x_train, y_train = np.array(x_tarin), np.array(y_train)

