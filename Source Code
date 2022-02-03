# By Harmeet Singh 

#--------------Crop Production Optimization using K-Means Clustering Algorithm--------------------

#------------- import libraries--------------------------------------------------------------------
from random import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('Crop_data.csv')
print(data.head())
print('\n')

#--------------Check for missing values------------------------------------------------------------
print("-------------Checking Null values if present--------------------")
print(data.isnull().sum())                     # No missing values in our Dataset
print('\n')


#-------------Exploratory Data Analysis-------------------------------------------------------------
#-------------Check the crops present in the dataset------------------------------------------------
print('------------Checking the different crops in the dataset & their counts----------')
print(data['label'].value_counts())
print('\n')

#-------------Check the average value of different parameters needed in the soil--------------------
print("--------Check the average value of different parameters needed in the soil-----------")
print("Average value of Nitrogen in the soil : {0:.2f}".format(data['N'].mean()))
print("Average value of Phosphorous in the soil : {0:.2f}". format(data['P'].mean()))
print("Average value of Potassium in the soil : {0:.2f}". format(data['K'].mean()))
print("Average value of Temperature in Celsius in the soil : {0:.2f}". format(data['temperature'].mean()))
print("Average value of Relative Humidity in the soil : {0:.2f}". format(data['humidity'].mean()))
print("Average value of ph in the soil : {0:.2f}".format(data['ph'].mean()))
print("Average value of rainfall in mm : {0:.2f}".format(data['rainfall'].mean()))

print("\n")

#-------------Check that which crops can be grown at above/below average conditions------------------
def compare(K):
    for i in K:
        print("-------------", i, "----------------------------","\n")
        print("Crops that require greater than average", i, '\n')
        print(data[ data[i] > data[i].mean()]['label'].unique())

        print("Crops that requirement less than average",i, "\n")
        print(data[data[i] <= data[i].mean()]['label'].unique())
        print("\n")

Parameter_list = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

compare(Parameter_list)

#------------Now we will cluster those crops which can be grown in simliar conditions-----------------
#------------For this we'll be using K-Means clustering algorithm-------------------------------------
#------------It is an un-supervised learning algo., which means we'll train our model with labels-----

from sklearn.cluster import KMeans

#-----------As it is an unsupervised learning algorithm, we will remove the labels from the data------
x = data.drop(['label'], axis = 1)
x = x.values

#-----------Here 'K' refers to the number of clusters we need-----------------------------------------
#-----------For this we'll plot "ELBOW Curve" to find the optimal value of K--------------------------

def elbow(k_values):
    Test_error = []
    for i in k_values:
        km = KMeans(n_clusters=i, random_state=0)
        km.fit(x)
        Test_error.append(km.inertia_)
    return Test_error

k_values = range(1,11)
Errors = elbow(k_values)

plt.plot(k_values, Errors)
plt.xlabel('Number of clusters')
plt.ylabel('Test Errors')
plt.title('Test Elbow Curve')
plt.show()

#------------We got K = 4, using Elbow curve---------------------------------------------------------
#------------Now we'll train our model using K = 4 --------------------------------------------------

model = KMeans(n_clusters=4, random_state=0, init = 'k-means++', )
y_means = model.fit_predict(x)
temp = data['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means,temp], axis = 1)
z = z.rename(columns = {0 : 'cluster'})

#------------Let's print our clusters now----------------------------------------------------------------

print("Crops in the First cluster :", z[ z['cluster'] == 0]['label'].unique())
print('\n')
print("Crops in the Second cluster :", z[ z['cluster'] == 1]['label'].unique())
print('\n')
print("Crops in the Third cluster :", z[ z['cluster'] == 2]['label'].unique())
print('\n')
print("Crops in the Fourth cluster :", z[ z['cluster'] == 3]['label'].unique())

#----------After clusters are ready, we need to split our dataset for predictive modeling----------------

X = data.drop(['label'], axis = 1)
Y = data['label']

#----------Train_Test Splitting and finally training the model--------------------------------------------

from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size= 0.25)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

from sklearn.metrics import classification_report
cr = classification_report(y_predict, y_test)
print(cr)

#---------Let's make the PREDICTION using our model-------------------------------------------------------

prediction = model.predict( (np.array([[90, 40, 40, 20, 80, 7, 200]])))
print(" The suggested crop for given climatic condition is : ", prediction)




















 

