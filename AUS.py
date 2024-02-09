#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[3]:


df=pd.read_csv("userbehaviour.csv")
df


# In[4]:


print("Lowest Screentime : ",df["Average Screen Time"].min())
print("Average Screentime : ",df["Average Screen Time"].mean())
print("Highest Screentime : ",df["Average Screen Time"].max())


# In[10]:


print("Least Amount Spent : ",df["Average Spent on App (INR)"].min())
print("Average Amount Spent : ",df["Average Spent on App (INR)"].mean())
print("Maximum Amount Spent : ",df["Average Spent on App (INR)"].max())


# In[11]:


sb.regplot(data=df,x="Average Screen Time",y="Average Spent on App (INR)")


# In[12]:


# Users who uninstalled the app had an average screen time of fewer than 5 minutes a day, and the average spent was less than 100. We can also see a linear relationship between the average screen time and the average spending of the users still using the app.
#Lets have a closer look to the relation between ratings and average screen time of the users
sb.regplot(data=df,x="Average Screen Time",y="Ratings")


# In[13]:


features = df[['userid', 'Average Screen Time', 'Average Spent on App (INR)','Left Review', 'Ratings', 'New Password Request', 'Last Visited Minutes']]
features_encoded = pd.get_dummies(features)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
kmeans_model = KMeans(n_clusters=3, random_state=42)
clusters= kmeans_model.fit_predict(scaled_features)
df["Segments"] = clusters
df.head()


# In[14]:


plt.figure(figsize=(8, 8))
sb.scatterplot(data=df, x='Last Visited Minutes', y='Average Spent on App (INR)', hue='Segments', palette='viridis', s=60, edgecolor='w')
plt.title('Scatter Plot of Last Visited Minutes vs. Average Spent on App (INR)')
plt.xlabel('Last Visited Minutes')
plt.ylabel('Average Spent on App (INR)')
plt.legend(title='Segments')
plt.show()


# In[ ]:




