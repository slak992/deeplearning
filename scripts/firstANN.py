import csv
import os
from math import remainder

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Sequential
from keras import layers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder



print(os.getcwd())
file_name = '../dataset/Churn_Modelling.csv'

#Create the datset
credit_df = pd.read_csv(file_name)

#create independant_variables and dependant variables
independant_variable = credit_df.iloc[:,3:-1]
dependant_variable = credit_df.iloc[:,-1]

print(independant_variable)
print(dependant_variable)

#Need to get the col, and row of each distict value in a dataframe
df_country_values = independant_variable['Geography'].drop_duplicates(keep='first')
country_names_index = {}
for each_country in df_country_values:
    row= independant_variable[independant_variable['Geography'] == each_country].index[0]
    col = independant_variable.columns.get_loc('Geography')
    country_names_index[each_country] = [row,col]
print(country_names_index)

df_gender_values = independant_variable['Gender'].drop_duplicates(keep='first')
gender_names_index = {}
for each_gender in df_gender_values:
    row=independant_variable[independant_variable['Gender'] == each_gender].index[0]
    col = independant_variable.columns.get_loc('Gender')
    gender_names_index[each_gender] = [row,col]
print(gender_names_index)

#encode the dataset - Lbel encoder, onehot encoding
lb_encoder = LabelEncoder()
independant_variable.iloc[:,2]=lb_encoder.fit_transform(independant_variable.iloc[:,2])

ct = ColumnTransformer(transformers=[('country_encoder',OneHotEncoder(),[1])],remainder='passthrough')
independant_variable = ct.fit_transform(independant_variable)
print(independant_variable)

country_encorder = ct.named_transformers_['country_encoder']
country_index = []
for index,country in enumerate(country_encorder.categories_[0]):
    country_index.append(index)
print(country_index)


for country in country_names_index:
    index_lst = []
    for index in country_index:
        index_lst.append(independant_variable[country_names_index[country][0], index])

    country_names_index[country].append(index_lst)
print(country_names_index)

for gender in gender_names_index:
    gender_names_index[gender].append([independant_variable[gender_names_index[gender][0],gender_names_index[gender][1]]])
print(gender_names_index)

#split the dataset

xtrain, xtest, ytrain, ytest = train_test_split(independant_variable,dependant_variable,test_size=0.2,random_state=0)

#feature scaling
scaler = StandardScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

#building the model
ann = Sequential()
ann.add(layers.Dense(6, activation='relu'))
ann.add(layers.Dense(6, activation='relu'))
ann.add(layers.Dense(1, activation='sigmoid'))


#compile
ann.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#train
ann.fit(xtrain,ytrain,epochs=20,batch_size=32)

data = []
data.append(country_names_index['France'][2])
data[0].extend([600,gender_names_index['Male'][2][0],40, 3, 60000, 2, 1, 1, 50000])
new_data_predict = ann.predict(scaler.transform(data))
print(new_data_predict)

if new_data_predict <= 0.5:
    print("Customer will stay")
else:
    print("Customer will not stay")

#test set evaluation
loss, accuracy = ann.evaluate(xtest,ytest)
print(loss)
print(accuracy)

y_pred = ann.predict(xtest)
print(y_pred)
y_pred_binary = (y_pred > 0.5).astype(int)
yred_arr = np.array(y_pred_binary).reshape(-1,1)
ytest_arr= np.array(ytest).reshape(-1,1)

print(np.concatenate((yred_arr,ytest_arr),1))

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(ytest_arr,yred_arr)
print(cm)
print(accuracy_score(ytest_arr,yred_arr))


