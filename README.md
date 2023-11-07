# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.choose the number of clusters

Decide how many clusters you want to identify in your data. This is a hyperparameter that you need to set in advance.

2.  Initialize cluster centroids

Randomly select K data points from your dataset as the initial centroids of the clusters.

3.  Assign data points to clusters

Calculate the distance between each data point and each centroid. Assign each data point to the cluster with the closest centroid. This step is typically done using Euclidean distance, but other distance metrics can also be used.

4.  Update cluster centroids

Recalculate the centroid of each cluster by taking the mean of all the data points assigned to that cluster.

5.  Repeat steps 3 and 4

Iterate steps 3 and 4 until convergence. Convergence occurs when the assignments of data points to clusters no longer change or change very minimally.

6.  Evaluate the clustering results

Once convergence is reached, evaluate the quality of the clustering results. This can be done using various metrics such as the within-cluster sum of squares (WCSS), silhouette coefficient, or domain-specific evaluation criteria.

7.   Select the best clustering solution

If the evaluation metrics allow for it, you can compare the results of multiple clustering runs with different K values and select the one that best suits your requirements

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Srihariharan S A
RegisterNumber: 212221040160


import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Mall_Customers.csv")
data.head()

data.info()

data.isnull().sum()


from sklearn.cluster import KMeans
wcss = []


for i in range(1,11):
  kmeans = KMeans(n_clusters = i, init = "k-means++")
  kmeans.fit(data.iloc[:, 3:])
  wcss.append(kmeans.inertia_)


plt.plot(range(1, 11), wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("wcss")
plt.title("Elbow Method")


km = KMeans(n_clusters = 5)
km.fit(data.iloc[:, 3:])


y_pred = km.predict(data.iloc[:, 3:])
y_pred


data["cluster"] = y_pred
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c = "red", label = "cluster0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c = "black", label = "cluster1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c = "blue", label = "cluster2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c = "green", label = "cluster3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c = "magenta", label = "cluster4")
plt.legend()
plt.title("Customer Segments")
  
*/
```

## Output:


data.head() function

![Screenshot 2023-11-03 160008](https://github.com/22008496/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119476113/fea28098-4822-440a-860e-0585c18c83c0)

 data.info()

![Screenshot 2023-11-03 160015](https://github.com/22008496/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119476113/7e731a47-0022-413e-a78e-0d8f0d085559)

 data.isnull().sum() function

![Screenshot 2023-11-03 160023](https://github.com/22008496/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119476113/30f81154-b064-4744-97d2-438195307d6f)

Elbow method Graph

![Screenshot 2023-11-03 160111](https://github.com/22008496/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119476113/648f40b1-ed16-4bde-8564-097108ee5796)

KMeans clusters

![Screenshot 2023-11-03 160218](https://github.com/22008496/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119476113/0fca4a8f-17f3-4117-be8b-eebc01880840)

Customer segments Graph

![Screenshot 2023-11-03 160229](https://github.com/22008496/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/119476113/a524438f-6360-4e75-9231-f17baa822e37)




## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
