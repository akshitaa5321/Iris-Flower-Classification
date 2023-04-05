#!/usr/bin/env python
# coding: utf-8

# # IRIS FLOWER CLASSIFICATION

# In[1]:


#importing neccesary libraries
import numpy as np # numercial python, arrays
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


#reading the dataset using pandas dataframe
data=pd.read_csv('Iris.csv')


# In[3]:


#fetch first 10 columns
data.head(10)


# In[4]:


data.tail(10)


# In[5]:


#Remove unnecessary feat from dataset Id
data.drop(columns=['Id'],axis=0,inplace=True)


# In[6]:


#Checking datatypes of each feat
data.dtypes


# In[7]:


data.shape
print('Rows:',data.shape[0])
print('Colums:',data.shape[1])


# In[8]:


data.size


# In[9]:


data.info


# In[10]:


data.describe()


# In[11]:


data.columns = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
df_split_iris=data.Species.str.split('-',n=-1,expand=True) #Remove prefix 'Iris-' from species col
df_split_iris.drop(columns=0,axis=1,inplace=True)#Drop 'Iris-' col
df_split_iris


# In[12]:


df=data.join(df_split_iris)
df


# In[13]:


df.rename({1:'species1'},axis=1,inplace=True) #Rename column
df


# In[14]:


df.drop(columns='Species',axis=1,inplace=True) #Drop excessive column


# In[15]:


df


# In[16]:


df.shape


# In[17]:


df.isna()


# In[18]:


#checking missing entries
df.isna().sum()


# In[19]:


df.describe()


# In[20]:


#categoriwise frequency of data
df.species1.value_counts()


# In[21]:


import matplotlib.pyplot as plt


# In[22]:


fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
l = ['Versicolor', 'Setosa', 'Virginica']
s = [50,50,50]
ax.pie(s, labels = l,autopct='%1.2f%%')
plt.show()


# In[23]:


import matplotlib.pyplot as plt
plt.figure(1)
plt.boxplot([df['SepalLength']])
plt.figure(2)
plt.boxplot([df['SepalWidth']])
plt.show()


# In[24]:


df.hist()
plt.show()


# In[25]:


df.plot(kind ='density',subplots = True, layout =(3,3),sharex = False)


# In[26]:


df.plot(kind ='box',subplots = True, layout =(2,5),sharex = False)


# In[27]:


import seaborn as sns


# In[28]:


plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='species1',y='PetalLength',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='species1',y='PetalWidth',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='species1',y='SepalLength',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='species1',y='SepalWidth',data=df)


# In[29]:


sns.pairplot(df,hue='species1')


# In[30]:


#Heat Maps
fig=plt.gcf()
fig.set_size_inches(10,7)
fig=sns.heatmap(df.corr(),annot=True,cmap='cubehelix',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)


# In[31]:


X = df['SepalLength'].values.reshape(-1,1)
print(X)


# In[32]:


Y = df['SepalWidth'].values.reshape(-1,1)
print(Y)


# In[33]:


plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.scatter(X,Y,color='Red')
plt.show()


# In[34]:


A = df['PetalLength'].values.reshape(-1,1)
print(A)


# In[35]:


B = df['PetalWidth'].values.reshape(-1,1)
print(B)


# In[36]:


plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.scatter(A,B,color='Red')
plt.show()


# In[37]:


#priting correlation
corr_mat = df.corr()
print(corr_mat)


# In[38]:


#models
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[39]:


#splitting into testing and training
train, test = train_test_split(df, test_size = 0.25)
print(train.shape)
print(test.shape)


# In[40]:


x_train = train[['SepalLength', 'SepalWidth', 'PetalLength',
                 'PetalWidth']]
y_train = train.species1

x_test = test[['SepalLength', 'SepalWidth', 'PetalLength',
                 'PetalWidth']]
y_test = test.species1


# In[41]:


x_train.head()


# In[42]:


y_train.head()


# In[43]:


x_test.head()


# In[44]:


y_test.head()


# In[45]:


#Using LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
prediction = model.predict(x_test)
print('Accuracy Score:',metrics.accuracy_score(prediction,y_test)*100)


# In[46]:


#Confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
confusion_mat = confusion_matrix(y_test,prediction)
print("Confusion matrix: \n",confusion_mat)
print(classification_report(y_test,prediction))


# In[47]:


#Using Support Vector
from sklearn.svm import SVC
model1 = SVC()
model1.fit(x_train,y_train)

pred = model1.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,pred)*100)


# In[48]:


#Using KNN Neighbors
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(x_train,y_train)
predict = model2.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,predict)*100)


# In[49]:


#Using Decision Tree
from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier(criterion='entropy',random_state=7)
model3.fit(x_train,y_train)
y_pred = model3.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred)*100)


# In[50]:


#Using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
model4=RandomForestClassifier(n_estimators=100)
model4.fit(x_train,y_train)
pred_y=model4.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred)*100)


# In[51]:


#Using GaussianNB
from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(x_train,y_train)
y_pred3 = model3.predict(x_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred3)*100)


# In[52]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','Support Vector Machines','KNN' ,'Decision Tree', 'Random Forest', 'Naive Bayes'],
    'Score': [100.0,100.0,97.36,100.0,100.0,97.36]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df.head(9)


# In[ ]:




