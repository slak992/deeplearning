import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
import numpy as np
data = pd.read_csv('../dataset/Credit_Card_Applications.csv')
x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

sc = MinMaxScaler(feature_range=(0,1))
x_scaled = sc.fit_transform(x)

#Training data
som = MiniSom(x=10,y=10,input_len=x_scaled.shape[1],sigma=1.0,learning_rate=0.5)
som.random_weights_init(x_scaled)
som.train_random(data=x_scaled,num_iteration=100)

from pylab import bone,pcolor,colorbar,plot,show
bone()
u_matrix = som.distance_map().T
pcolor(u_matrix)
colorbar()
markers = ['o','s']
colors = ['r','g']
fraud_cordinates = set()
threshold = np.percentile(u_matrix,95)
for i,val in enumerate(x_scaled):
    w = som.winner(val)
    is_outlier = u_matrix[w[1], w[0]] > threshold  # Check high distance in transposed matrix
    if is_outlier and colors[y[i]] == 'g':  # Check if it's a green square
        fraud_cordinates.add(w)
    plot(
        w[0]+0.5,
        w[1]+0.5,
        markers[y[i]],
        markeredgecolor=colors[y[i]],
        markerfacecolor='None',
        markeredgewidth=1,
        markersize=10
    )
show()
print(fraud_cordinates)
mappings = som.win_map(x_scaled)
fraud_data = tuple(mappings[cord] for cord in fraud_cordinates)
frauds_scaled = np.concatenate(fraud_data,axis=0)
fraud_details = sc.inverse_transform(frauds_scaled)
print(fraud_details)








