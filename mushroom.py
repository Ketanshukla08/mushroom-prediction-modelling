import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


#importing the csv data
df= pd.read_csv("C:/Users/KETANLA/Desktop/mushroom_train.csv")
tf=pd.read_csv("C:/Users/KETANLA/Desktop/mushroom_test.csv")


#definig the test set and train set data
X_train= df.ix[:,1::]
X_test= tf
Y_train= df.ix[:,0]



#joining the input set of both question and train
frames= [X_train, X_test]
X=pd.concat(frames, ignore_index= True)

#visualizing the columns and its data tpye
X.head()

#now converting the string values to numerical values
for i in df.columns:
    if i!='radius' and i!='weight' and i!='class':
        X[i]=X[i].astype("category",ordered=True, categories=X[i].unique()).cat.codes
    df[i]=df[i].astype("category",ordered=True, categories=df[i].unique()).cat.codes

#now checking the correlation beetween the sets
df.corr()


#due to constant value of column veil-type
#we are getting NaN in correlation,so removing that input column
del X['veil-type']


#there are only two continuos value holding features, so lets visualize the distribution of these data
sns.distplot(X.radius)
sns.distplot(X.weight)


#we should normalize these features before fitting in our algo
#normalizing the continuos values only, by creating a separate table for these two features
X_n=pd.DataFrame(X.ix[:,['radius','weight']])

rank_mean = X_n.stack().groupby(X_n.rank(method='first').stack().astype(int)).mean()
X_n=X_n.rank(method='min').stack().astype(int).map(rank_mean).unstack()



#now amending these values to 'X' set
X.radius=X_n.ix[:,0]
X.weight=X_n.ix[:,1]


#visualizing the distribution now
sns.distplot(X.radius)
sns.distplot(X.weight)
#distribution is better now


#now spliting the train and test set within the Mushroom_train data we are given
#here i have splitted manually (for no reason)
#we can use train_test_split library to have this splitted in any fraction we want
trainX=X.ix[0:4800,:]
trainY=Y_train.ix[0:4800]
testX= X.ix[4801:5685,:]
testY= Y_train.ix[4801:5685]

#now checking the accuracy on logistic regression(for self satisfaction :p)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR=LR.fit(trainX,trainY)
LR.score(testX,testY)
#here i got accuracy of 95.5%(increased due to normalization)

#calling xgboost 
from xgboost import XGBClassifier
myprediction = XGBClassifier()

#fitting the training data
myprediction.fit(trainX, trainY)

#now checking the accuracy on testX

Y_pred = myprediction.predict(testX)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(testY, Y_pred)
print (accuracy)
#accuracy is 100%(strangely)

#with this much accuracy, no need to further amend, so final ouput
#now finally predicting the output for mushroom_test data
Y_out= myprediction.predict(X.ix[5686::,:])
#Y_out is our required prediction
classes= Y_out
print(classes)
