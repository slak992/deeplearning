import pandas as pd

from scripts.firstANN import y_pred

customers_df = pd.read_csv("../dataset/Credit_Card_Applications.csv")
X = customers_df.iloc[:,:-1].values
Y = customers_df.iloc[:,-1].values

#Data Preprocessing - Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
x_scaled = sc.fit_transform(X)

#Training SOM
from minisom import MiniSom
som = MiniSom(x=10,y=10,input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(x_scaled)
som.train_random(data=x_scaled,num_iteration=100)

#Plotting the clusters
from pylab import bone,pcolor,colorbar,plot,show
import numpy as np

bone()
u_matrix = som.distance_map().T
pcolor(u_matrix)
colorbar()
outlier_thrshold = np.percentile(u_matrix,90)
markers = ['o','s']
colores = ['r','g']
outlier_cord = set()
for i,data in enumerate(x_scaled):
    w = som.winner(data)
    is_outlier = u_matrix[w[1],w[0]] > outlier_thrshold
    if is_outlier and colores[Y[i]] == 'g':
        outlier_cord.add(w)
    plot(
        w[0]+0.5,
        w[1]+0.5,
        markers[Y[i]],
        markeredgecolor=colores[Y[i]],
        markerfacecolor = 'None',
        markeredgewidth=1,
        markersize=10
    )
show()

mapping = som.win_map(x_scaled)
outliers = tuple(mapping[outlier] for outlier in outlier_cord)
outliers_data = np.concatenate(outliers,axis=0)
fraud_customers = sc.inverse_transform(outliers_data)
print(fraud_customers)
#Integration with ANN

features = customers_df.iloc[:,1:].values
from sklearn.preprocessing import StandardScaler
scc = StandardScaler()
feature_scaled = scc.fit_transform(features)

is_fraud = np.zeros(len(features))

for i in range(len(features)):
    if customers_df.iloc[i,0] in fraud_customers:
        is_fraud[i] = 1
print("Done with the data preparation")

#ANN model preparation

from keras import Sequential, layers

ann_classifier = Sequential()
ann_classifier.add(layers.Dense(units=2,activation='relu',kernel_initializer='uniform',input_dim=15))
ann_classifier.add(layers.Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))
ann_classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
ann_classifier.fit(feature_scaled,is_fraud,batch_size=1,epochs=2)
y_pred = ann_classifier.predict(feature_scaled)
final_prediction = np.concatenate((customers_df.iloc[:,0:1].values,y_pred),axis=1)
final_prediction_sorted = final_prediction[final_prediction[:,1].argsort()]
print(final_prediction_sorted)
reverse_order_prediction = final_prediction_sorted[::-1]
print(reverse_order_prediction)