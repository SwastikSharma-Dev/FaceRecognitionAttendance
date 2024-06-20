# SWASTIK SHARMA 102203231
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
data_df = pd.read_csv("data.csv")
#test_df = pd.read_csv("test.csv")
data_df.head()
data_df.shape
y=data_df['label']
x=data_df.drop('label',axis=1)
#x_for_test_data=test_df[:]
type(x)
plt.figure(figsize=(7,7))
some_digit=1266
some_digit_image = x.iloc[some_digit].to_numpy()
plt.imshow(np.reshape(some_digit_image, (28,28)))
print(y[some_digit])
sns.countplot( x='label', data=data_df)
from sklearn.model_selection import  train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state = 40)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
#from sklearn.preprocessing import  StandardScaler
#scaler = StandardScaler()
#scaler.fit(x_train,y_train)
#x_train = scaler.transform(x_train)
#x_train.shape
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 10)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
print(accuracy_score(y_test, y_pred))
print("------------------------------------")
print("Classification Report")
print(classification_report(y_test, y_pred))
print("------------------------------------")
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("------------------------------------")
#y_pred_on_test_data = classifier.predict(x_for_test_data)
#y_pred_on_test_data