import pandas as pd
import datetime
import matplotlib.pyplot as plt
import re
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
data=pd.read_excel("C:/Users/Human/Downloads/CreditAnalysis_data.xlsx")
data.isna().sum()
data.info()
data.columns
data.dropna(axis=0,inplace=True)
data.drop(data.loc[(data['master_order_status'] =='Partially Delivered') |(data['master_order_status']=='cancelled') |(data['master_order_status'] =='rejected')].index,inplace=True)
#parially deliverd
#rejected

data['date']=pd.to_datetime(data['created']).dt.date
a=data.groupby(['retailer_names'],as_index=False)['bill_amount'].max()
a.columns
data['retailer_names'].value_counts()
a.sort_values('bill_amount',ascending=False,inplace=True)
#grouping
recency_table=data.groupby(['retailer_names'],as_index=False)['date'].max()
recency=recency_table['date'].max()-recency_table['date']
recency_table['recency']=recency.dt.days
plt.bar(a['retailer_names'].head(),a['bill_amount'].head())

i=[]
for i in recency_table['retailer_names']:
    p=re.findall('[0-9]+', str(i))
    id.extend(list(p))
recency_table['id']=id
recency_table.drop(labels=['date'],axis=1,inplace=True)
recency_table

plt.boxplot(recency_table['recency'])
recency_table['a'] = np.where(recency_table['recency']<=recency_table['recency'].quantile(0.25),1,np.where(recency_table['recency']<=recency_table['recency'].quantile(0.50),2,np.where(recency_table['recency']<=recency_table['recency'].quantile(0.75),3,4)))
#kmeans cluster
# x=[]
# y=[]
# for i in range(2,10):
#     cl=KMeans(n_clusters=i).fit(recency_table[['recency']])
#     x.append(i)
#     y.append(cl.inertia_)
# plt.plot(x,y)
# cl=KMeans(n_clusters=4).fit(recency_table[['recency']])
# center=cl.cluster_centers_
# recency_table['labels']=cl.labels_
#frequency
frequence=data.groupby('retailer_names',as_index=False)['date'].count()
frequence.rename(columns={'date':'number_of_transaction'},inplace=True)
frequence.columns
#for frequency cluster
plt.boxplot(frequence['number_of_transaction'])
frequence['b']=np.where(frequence['number_of_transaction']<=frequence['number_of_transaction'].quantile(0.25),1,np.where(frequence['number_of_transaction']<=frequence['number_of_transaction'].quantile(0.50),2,np.where(frequence['number_of_transaction']<=frequence['number_of_transaction'].quantile(0.75),3,4)))
# x=[]
# y=[]
# for i in range(2,10):
#     km=KMeans(n_clusters=i).fit(frequence[['number_of_transaction']])
#     x.append(i)
#     y.append(km.inertia_)
# plt.plot(x,y)
# fr=KMeans(n_clusters=4).fit(frequence[['number_of_transaction']])
# fr.cluster_centers_
# frequence['cluster']=fr.labels_
#monitary value
revenue=data.groupby('retailer_names',as_index=False)['value'].sum()
revenue.columns
#revanue clusters
plt.boxplot(revenue['value'])
revenue['c']=np.where(revenue['value']<=revenue['value'].quantile(0.25),1,np.where(revenue['value']<=revenue['value'].quantile(0.50),2,np.where(revenue['value']<=revenue['value'].quantile(0.75),3,4)))
# x=[]
# y=[]
# for i in range(2,10):
#     km=KMeans(n_clusters=i).fit(revenue[['value']])
#     x.append(i)
#     y.append(km.inertia_)
# plt.plot(x,y)
# rv=KMeans(n_clusters=5).fit(revenue[['value']])
# rv.cluster_centers_
# revenue['cluster']=rv.labels_
over_all_score=pd.DataFrame({'name':revenue['retailer_names'],'recency':recency_table['recency'],'recency_cluster':recency_table['a'],'frequence':frequence['number_of_transaction'],'frequency_cluster':frequence['b'],'revenue':revenue['value'],'revenue_clust':revenue['c'],'score':revenue['c']+frequence['b']+recency_table['a']})
over_all_score['score'].max()
over_all_score['score'].value_counts()

over_all_score['rank']=np.where(over_all_score['score']<=3,0,np.where(over_all_score['score']<=6,1,2))


#creating noise
noice=data.groupby('retailer_names',as_index=False)['dist_names'].max()
from sklearn.preprocessing import LabelEncoder
over_all_score['noise']=LabelEncoder().fit_transform(noice['dist_names'])
over_all_score['name']=over_all_score['name'].str.replace('RetailerID','')
over_all_score.info()
over_all_score['name']=over_all_score['name'].astype('int64')
x=over_all_score.drop(labels=['rank','noise'],axis=1)
y=over_all_score['rank']


#log=sm.logit('rank ~ recency + recency_cluster + frequence + frequency_cluster + revenue + revenue_clust + score',data=over_all_score).fit()
x.columns
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier
def pred(i):
    dt=DecisionTreeClassifier(criterion = 'entropy')
    dt.fit(x_train, y_train)
    test_res=dt.predict([i])
    return test_res;
o=[1,0,1,198,4,4426,4,9]
pred(o)
dt=DecisionTreeClassifier(criterion = 'entropy')
dt.fit(x_train, y_train)
import pickle
pickle.dump(dt,open('model.pkl','wb'))

