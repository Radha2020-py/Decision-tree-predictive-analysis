#!/usr/bin/env python
# coding: utf-8

# # Predict party affiliation based on voting record
# 

# In[1]:


#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

#Surpress the warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


#Import application_data.csv
data = pd.read_csv('House84.csv')


# In[3]:


#Explore datasets - application_data.csv - shape
data.shape


# In[4]:


#Explore datasets - application_data.csv head
data.head()


# In[5]:


data.info()


# In[6]:


data.columns


# # Inspect Null values

# In[7]:


pointer=len(data)
data.isnull().sum()


# In[8]:


data.isnull().sum(axis=1)


# In[9]:


data.describe()


# In[10]:


#All the variables are categorical Lets examine them
data['Target'] = np.where(data['Class Name'] == 'democrat', 1, 0)
data.Target.value_counts()


# We have 61 percent of  Democratic Party 

# In[11]:


plt.figure(figsize = (10,5))
sns.set(style="darkgrid")
ax = sns.countplot(x='Class Name',data=data)
ax.set_xticklabels(ax.get_xticklabels(), fontsize = 30)
plt.tight_layout()


# This imbalance is fair enough. But the size of the dataset is pretty small so it might cause some bias at the end but lets see

# # handicapped-infants

# In[12]:


data[' handicapped-infants'].value_counts()


# In[13]:


plt.figure(figsize = (10,5))
sns.set(style="darkgrid")
ax = sns.countplot(x=' handicapped-infants',data=data)
ax.set_xticklabels(ax.get_xticklabels(), fontsize = 30)
plt.tight_layout()


# Lets see what are the chances of them being belonging to Democratic Party

# In[14]:


data.groupby(' handicapped-infants')['Target'].mean().sort_values(ascending = False)

So this varible presumes that the chances of people with handicapped infants of belonging to Democrats are 83 percent
# In[15]:


data['handicapped_infants_n'] = np.where(data[' handicapped-infants'] == 'n', 1, 0)
data['handicapped_infants_y'] = np.where(data[' handicapped-infants'] == 'y', 1, 0)


# # water-project-cost-sharing

# In[16]:


data[' water-project-cost-sharing'].value_counts()


# In[17]:


data.groupby(' water-project-cost-sharing')['Target'].mean().sort_values(ascending = False)


# In[18]:


data['water-project-cost-sharing_n'] = np.where(data[' water-project-cost-sharing'] == 'n', 1, 0)
data['water-project-cost-sharing_y'] = np.where(data[' water-project-cost-sharing'] == 'y', 1, 0)


# # adoption-of-the-budget-resolution

# In[19]:


data[' adoption-of-the-budget-resolution'].value_counts()


# In[20]:


data.groupby(' adoption-of-the-budget-resolution')['Target'].mean().sort_values(ascending = False)


# In[21]:


data['adoption-of-the-budget-resolution_n'] = np.where(data[' adoption-of-the-budget-resolution'] == 'n', 1, 0)
data['adoption-of-the-budget-resolution_y'] = np.where(data[' adoption-of-the-budget-resolution'] == 'y', 1, 0)


# # physician-fee-freeze

# In[22]:


data[' physician-fee-freeze'].value_counts()


# In[23]:


data.groupby(' physician-fee-freeze')['Target'].mean().sort_values(ascending = False)


# In[24]:


data['physician-fee-freeze_n'] = np.where(data[' physician-fee-freeze'] == 'n', 1, 0)
data['physician-fee-freeze_y'] = np.where(data[' physician-fee-freeze'] == 'y', 1, 0)


# In[25]:


data.columns


# In[26]:


cols_to_be_dropped = ['Class Name', ' handicapped-infants', ' water-project-cost-sharing',
       ' adoption-of-the-budget-resolution', ' physician-fee-freeze',
       ' el-salvador-aid', ' religious-groups-in-schools',
       ' anti-satellite-test-ban', ' aid-to-nicaraguan-contras', ' mx-missile',
       ' immigration', ' synfuels-corporation-cutback', ' education-spending',
       ' superfund-right-to-sue', ' crime', ' duty-free-exports',
       ' export-administration-act-south-africa']

#We will drop these cols at the end of our analysis


# # el-salvador-aid

# In[27]:


data[' el-salvador-aid'].value_counts()


# In[28]:


data.groupby(' el-salvador-aid')['Target'].mean()


# In[29]:


data['el-salvador-aid_n'] = np.where(data[' el-salvador-aid'] == 'n', 1, 0)
data['el-salvador-aid_y'] = np.where(data[' el-salvador-aid'] == 'y', 1, 0)


# # religious-groups-in-schools

# In[30]:


data[' religious-groups-in-schools'].value_counts()


# In[31]:


data.groupby(' religious-groups-in-schools')['Target'].mean()


# In[32]:


data['religious-groups-in-schools_n'] = np.where(data[' religious-groups-in-schools'] == 'n', 1, 0)
data['religious-groups-in-schools_y'] = np.where(data[' religious-groups-in-schools'] == 'y', 1, 0)


# # anti-satellite-test-ban

# In[33]:


data[' anti-satellite-test-ban'].value_counts()


# In[34]:


data.groupby(' anti-satellite-test-ban')['Target'].mean()


# In[35]:


data['anti-satellite-test-ban_n'] = np.where(data[' anti-satellite-test-ban'] == 'n', 1, 0)
data['anti-satellite-test-ban_y'] = np.where(data[' anti-satellite-test-ban'] == 'y', 1, 0)


# # aid-to-nicaraguan-contras

# In[36]:


data[' aid-to-nicaraguan-contras'].value_counts()


# In[37]:


data.groupby(' aid-to-nicaraguan-contras')['Target'].mean()

# Massively favored by dems
# In[38]:


data['aid-to-nicaraguan-contras_n'] = np.where(data[' aid-to-nicaraguan-contras'] == 'n', 1, 0)
data['aid-to-nicaraguan-contras_y'] = np.where(data[' aid-to-nicaraguan-contras'] == 'y', 1, 0)


# # Mx-missile

# In[39]:


data[' mx-missile'].value_counts()


# In[40]:


data.groupby(' mx-missile')['Target'].mean()


# In[41]:


data['mx-missile_n'] = np.where(data[' mx-missile'] == 'n', 1, 0)
data['mx-missile_y'] = np.where(data[' mx-missile'] == 'y', 1, 0)


# # Immigration

# In[42]:


data[' immigration'].value_counts()


# In[43]:


data.groupby(' immigration')['Target'].mean()


# In[44]:


data['immigration_n'] = np.where(data[' immigration'] == 'n', 1, 0)
data['immigration_y'] = np.where(data[' immigration'] == 'y', 1, 0)


# # Synfuels-corporation-cutback

# In[45]:


data[' synfuels-corporation-cutback'].value_counts()


# In[46]:


data.groupby(' synfuels-corporation-cutback')['Target'].mean()


# In[47]:


data['synfuels-corporation-cutback_n'] = np.where(data[' synfuels-corporation-cutback'] == 'n', 1, 0)
data['synfuels-corporation-cutback_y'] = np.where(data[' synfuels-corporation-cutback'] == 'y', 1, 0)


# # Education-spending

# In[48]:


data[' education-spending'].value_counts()


# In[49]:


data.groupby(' education-spending')['Target'].mean()


# In[50]:


data['education-spending_n'] = np.where(data[' education-spending'] == 'n', 1, 0)
data['education-spending_y'] = np.where(data[' education-spending'] == 'y', 1, 0)


# # Superfund-right-to-sue

# In[51]:


data[' superfund-right-to-sue'].value_counts()


# In[52]:


data.groupby(' superfund-right-to-sue')['Target'].mean()


# In[53]:


data['superfund-right-to-sue_n'] = np.where(data[' superfund-right-to-sue'] == 'n', 1, 0)
data['superfund-right-to-sue_y'] = np.where(data[' superfund-right-to-sue'] == 'y', 1, 0)


# # Crime 

# In[54]:


data[' crime'].value_counts()


# In[55]:


data.groupby(' crime')['Target'].mean()


# In[56]:


data['crime_n'] = np.where(data[' crime'] == 'n', 1, 0)
data['crime_y'] = np.where(data[' crime'] == 'y', 1, 0)


# # Duty-free-exports

# In[57]:


data[' duty-free-exports'].value_counts()


# In[58]:


data.groupby(' duty-free-exports')['Target'].mean()


# In[59]:


data['duty-free-exports_n'] = np.where(data[' duty-free-exports'] == 'n', 1, 0)
data['duty-free-exports_y'] = np.where(data[' duty-free-exports'] == 'y', 1, 0)


# # Export-administration-act-south-africa

# In[60]:


data[' export-administration-act-south-africa'].value_counts()


# In[61]:


data.groupby(' export-administration-act-south-africa')['Target'].mean()


# In[62]:


data['export-administration-act-south-africa_n'] = np.where(data[' export-administration-act-south-africa'] == 'n', 1, 0)
data['export-administration-act-south-africa_y'] = np.where(data[' export-administration-act-south-africa'] == 'y', 1, 0)


# In[63]:


data = data.drop(cols_to_be_dropped, axis = 1)
data.head()


# In[64]:


data.shape


# In[65]:


data.columns


# # Decision Tree 

# In[66]:


from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# In[67]:


features = ['Target', 'handicapped_infants_n', 'handicapped_infants_y',
       'water-project-cost-sharing_n', 'water-project-cost-sharing_y',
       'adoption-of-the-budget-resolution_n',
       'adoption-of-the-budget-resolution_y', 'physician-fee-freeze_n',
       'physician-fee-freeze_y', 'el-salvador-aid_n', 'el-salvador-aid_y',
       'religious-groups-in-schools_n', 'religious-groups-in-schools_y',
       'anti-satellite-test-ban_n', 'anti-satellite-test-ban_y',
       'aid-to-nicaraguan-contras_n', 'aid-to-nicaraguan-contras_y',
       'mx-missile_n', 'mx-missile_y', 'immigration_n', 'immigration_y',
       'synfuels-corporation-cutback_n', 'synfuels-corporation-cutback_y',
       'education-spending_n', 'education-spending_y',
       'superfund-right-to-sue_n', 'superfund-right-to-sue_y', 'crime_n',
       'crime_y', 'duty-free-exports_n', 'duty-free-exports_y',
       'export-administration-act-south-africa_n',
       'export-administration-act-south-africa_y']


# In[68]:


train, test = train_test_split(data, test_size = 0.35, random_state = 42)


# In[69]:


train.shape, test.shape


# In[70]:


X_train = train.drop('Target', axis = 1)
X_test = test.drop('Target', axis = 1)
y_train = train['Target']
y_test = test['Target']


# In[71]:


X_train = train[features]
Y_train = train['Target']


# In[72]:


X_test = test[features]
Y_test = test['Target']


# In[73]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[74]:


from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import pydotplus
import graphviz
import io 
from scipy import misc


# In[75]:


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = features,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Predict party.png')
Image(graph.create_png())


# In[76]:


# Create Decision Tree classifer object
clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# In[77]:


dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True, feature_names = features,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('Predict party.png')
Image(graph.create_png())


# In[78]:


from sklearn.tree import DecisionTreeClassifier
DTC = DecisionTreeClassifier()


# In[79]:


from sklearn.model_selection import cross_val_score
accuraccies = cross_val_score(estimator = DTC, X= train, y=train, cv=10)
print("Average Accuracies: ",np.mean(accuraccies))
print("Standart Deviation Accuracies: ",np.std(accuraccies))


# In[80]:


DTC.fit(train,train) #learning
#prediciton
print("Decision Tree Score: ",DTC.score(test,test))
DTCscore = DTC.score(test,test)


# # Random Forest Classifier

# In[81]:


from sklearn.ensemble import RandomForestClassifier


# # Gini

# In[95]:


def gini(list_of_values):
    sorted_list = sorted(list_of_values)
    height, area = 0, 0
    for value in sorted_list:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area


# # Confusion Matrix

# In[96]:


def plot_confusion_matrix(y_true, y_pred, title = 'Confusion matrix', cmap=plt.cm.Blues):
    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    print ('Classification Report:\n')
    print (classification_report(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    def plot_confusion_matrix_plot(cm, title = 'Confusion matrix', cmap=plt.cm.Blues):
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(y_test.unique()))
        plt.xticks(tick_marks, rotation=45)
        plt.yticks(tick_marks)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
    print (cm)
    plot_confusion_matrix_plot(cm=cm)


# In[97]:


rf = RandomForestClassifier(criterion = 'gini', 
                            max_depth = 8,
                            max_features = 'auto',
                            min_samples_leaf = 0.01, 
                            min_samples_split = 0.01,
                            min_weight_fraction_leaf = 0.0632, 
                            n_estimators = 1000,
                            random_state = 50, 
                            warm_start = False)


# In[98]:


rf.fit(X_train, y_train)


# In[99]:


pred = rf.predict(X_test)


# In[100]:


from sklearn.metrics import accuracy_score, confusion_matrix


# In[101]:


accuracy_score(pred, y_test)*100


# In[102]:


plot_confusion_matrix(y_test, pred)


# In[103]:


predicted_probs = rf.predict_proba(X_test)


# In[104]:


gini(predicted_probs[:,1])


# In[105]:


predicted_train = rf.predict(X_train)


# In[106]:


accuracy_score(predicted_train, y_train)


# In[107]:


plot_confusion_matrix(y_train, predicted_train)


# Only 7 wrong predictions
# 
# 
# 4 flase positives and 3 false negatives
# 
# 
# False postives are the ones that were actually democrats but predocted as republic
# 
# 
# False negatives are the ones that were actually republican but were predicted as dems

# In[ ]:




