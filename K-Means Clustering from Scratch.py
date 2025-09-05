import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import random
import math

#read csv
df = pd.read_csv("city_data_50.csv")
data_list = df.values.tolist()

df = df[["latitude", "longitude"]].copy()
df.columns = ["f1", "f2"]          

data_list = df.values.tolist()
'''
print(data_list)

'''

#plot
sns.scatterplot(x='f1', y='f2', data=df)
plt.title('Before k-means clustering')
plt.xlabel('F1')
plt.ylabel('F2')
plt.show()



def centroids(k):
    #pick first k points as centroid
    centroid = data_list[:k]

    '''
    print('Centroids:', centroid)
    '''
    return centroid



def calculate_distance(point, centroids):
    distances = []

    #use the formula to get distances
    for centroid in centroids:
        distance = math.sqrt((centroid[0] - point[0])**2 + (centroid[1] - point[1])**2)
        distances.append(distance)
        
    '''
    print(distances)
    '''   
    return distances



def assign_points(distances):
    cluster_one = []
    cluster_two = []

    #get the point with the miminum distance
    for i, distance in enumerate(distances):
        if distance[0] == min(distance):
            cluster_one.append(data_list[i])
        if distance[1] == min(distance):
            cluster_two.append(data_list[i])
    
    #update centroid with new data
    update_centroid(cluster_one)
    update_centroid(cluster_two)

    '''
    print("First cluster list:", cluster_one)
    print("Second cluster list:", cluster_two)
    '''
    return cluster_one, cluster_two



def update_centroid(cluster):
    #initializes list of 0 as centroid
    cluster_length = len(cluster)
    index = len(cluster[0])
    updated_centroid = [0] * index

    new_centroid = []

    #add the elements
    for point in cluster:
        for i in range(index):
            updated_centroid[i] += point[i]

    #average and update centroid
    for i in updated_centroid:
        j = i / cluster_length
        new_centroid.append(j)
        updated_centroid = new_centroid

    '''
    print('Updated centroids:', updated_centroid)
    '''
    return updated_centroid



def kmeans(k):
    centroids_list = centroids(k)
    distances = []
    cluster_one = []
    cluster_two = []
    
    #initial centroids
    for point in data_list:
        point_distances = calculate_distance(point, centroids_list)
        distances.append(point_distances)
    
    #loop untill centroids match
    while True:
        new_cluster_one, new_cluster_two = assign_points(distances)
        
        #update to new centroids
        new_centroid_one = update_centroid(new_cluster_one)
        new_centroid_two = update_centroid(new_cluster_two)
        
        #check if old and new centroids are same
        if new_centroid_one == centroids_list[0] and new_centroid_two == centroids_list[1]:
            break
        
        #update to new data
        centroids_list = [new_centroid_one, new_centroid_two]
        cluster_one = new_cluster_one
        cluster_two = new_cluster_two
        distances = []
        
        #recalculate 
        for point in data_list:
            point_distances = calculate_distance(point, centroids_list)
            distances.append(point_distances)

    #print results
    cluster_one_size = len(cluster_one)
    cluster_two_size = len(cluster_two)
    print("Cluster 1 data:")
    print("Size of Cluster:", cluster_one_size)
    print("Centroid:", centroids_list[0])
    print("---------------------------------------------")
    print("Cluster 2 data:")
    print("Size of Cluster:", cluster_two_size)
    print("Centroid:", centroids_list[1])

    #plot results
    sns.scatterplot(x='f1', y='f2', data=df)

    for point in cluster_one:
        plt.scatter(point[0], point[1], color='red')
    for point in cluster_two:
        plt.scatter(point[0], point[1], color='blue')

    plt.scatter(centroids_list[0][0], centroids_list[0][1], marker='X', color='black', label='Centroid 1')
    plt.scatter(centroids_list[1][0], centroids_list[1][1], marker='X', color='black', label='Centroid 2')
    plt.title('After k-means clustering')
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.show()
    return



#function call
k = 2
kmeans(k)

