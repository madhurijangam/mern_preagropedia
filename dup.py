from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import tree
import warnings
warnings.filterwarnings('ignore')

merge_fert = pd.read_csv("dataset.csv")
merge_fert
features = merge_fert[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
target = merge_fert['label']
#features = df[['temperature', 'humidity', 'ph', 'rainfall']]
labels = merge_fert['label']
acc = []
model = []
from sklearn.model_selection import train_test_split
Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state=2)

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(Xtrain,Ytrain)

predicted_values = RF.predict(Xtest)

x = metrics.accuracy_score(Ytest, predicted_values)
acc.append(x)
model.append('RF')
print("RF's Accuracy is: ", x)





#print(classification_report(Ytest,predicted_values))
#data = np.array([[104,18, 30, 23.603016, 60.3, 6.7, 140.91]])
#prediction = RF.predict(data)
#print(prediction)




# save a pickle
import pickle
# Dump the trained Naive Bayes classifier with Pickle
RF_pkl_filename = 'RandomForest.pkl'
# Open the file to save as pkl file
RF_Model_pkl = open(RF_pkl_filename, 'wb')
pickle.dump(RF, RF_Model_pkl)
# Close the pickle instances
RF_Model_pkl.close()






