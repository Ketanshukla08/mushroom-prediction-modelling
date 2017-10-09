import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
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

#now converting the string values to numerical values of df dataframe
for i in df.columns:
    df[i]=df[i].astype("category",ordered=True, categories=df[i].unique()).cat.codes

#now checking the correlation beetween the sets
df.corr()

# HERE WE CAN SEE THAT 'CLASS' DOES NOT DEPEND ON SOME FEATURES EFFECTIVELY, SO WE WOULD REMOVE THOSE FEATURES
del X['cap-shape']
del X['cap-color']
del X['radius']
del X['weight']
del X['veil-type']
del X['stalk-color-above-ring']
del X['stalk-color-below-ring']
del X['ring-type']
del X['habitat']

del df['cap-shape']
del df['cap-color']
del df['radius']
del df['weight']
del df['veil-type']
del df['stalk-color-above-ring']
del df['stalk-color-below-ring']
del df['ring-type']
del df['habitat']


#checking the data via stripplot, to get to know the categories of features,
fig, ax = plt.subplots()
fig.set_size_inches(17, 10)
sns.stripplot(data=df)

#now checking the data distribution
for i in df.columns:
    sns.distplot(df[i])
    plt.show()

#here we can see from distribution plot that gill-attachment, gill-spacing and ring-number features have almost constant value over their columns
#so we can remove these columns

del X['gill-spacing']
del X['gill-attachment']
del X['ring-number']

#now making the dummies of categorical features to impliment logistic regression
#it will give numerical value(1 or 0) coressponding to true or false for each category of every feature
X=pd.get_dummies(X)
X.head()

#now spliting the train and test set within the Mushroom_train data we are given
#here i have splitted manually (for no reason)
#we can use train_test_split library to have this splitted in any fraction we want
trainX=X.ix[0:4800,:]
trainY=Y_train.ix[0:4800]
testX= X.ix[4801:5685,:]
testY= Y_train.ix[4801:5685]

#now checking the accuracy on logistic regression

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR=LR.fit(trainX,trainY)
print(LR.score(testX,testY))

#we got accuracy of 100% for this set
#with this much accuracy, no need to further amend, so final ouput
#now finally predicting the output for mushroom_test data
Y_out= LR.predict(X.ix[5686::,:])
#Y_out is our required prediction
classes= pd.DataFrame(Y_out)
#print(classes)
