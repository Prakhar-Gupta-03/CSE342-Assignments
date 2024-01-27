import numpy as np
import matplotlib.pyplot as plt

def initialize_centroids(data, k):
    # Randomly initialising the centroids to the k centers
    # The first k rows of data are initialised as the centroids
    centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        centroids[i,0] = data[i,0]
        centroids[i,1] = data[i,1]
    return centroids

def k_means_algorithm(data, k, max_iteration):
    centroids = initialize_centroids(data, k)
    # initialize the cluster assignments
    cluster_assignments = np.zeros(data.shape[0])
    # initialize the previous cluster assignments
    prev_cluster_assignments = np.ones(data.shape[0])
    # while loop breaking condition
    break_loop = False
    iteration = 0
    # storing the cluster to which each data point is assigned
    # loop until the cluster assignments converge
    # the cluster assignments converge when prev_cluster_assignments == cluster_assignments, i.e. when the cluster assignments do not change from one iteration to the next
    while (break_loop==False and iteration<max_iteration):
        # calculate the distance from each data point to each centroid
        distances = np.zeros((data.shape[0], k))
        for i in range(k):
            distances[:,i] = np.linalg.norm(data - centroids[i,:], axis=1)
        # assign each data point to the closest centroid
        prev_cluster_assignments = cluster_assignments.copy()
        # cluster_assignments = np.argmin(distances, axis=1)
        cluster_assignments = np.zeros(data.shape[0])
        # traverse over each row of the distances matrix and assign the data point to the closest centroid
        for i in range(data.shape[0]):
            # traverse over the distances of the data point for each centroid and find the centroid with the minimum distance
            min_distance = distances[i,0]               # initializing the minimum distance to the distance of the data point to the first centroid
            cluster_index = 0                           # cluster index keeps track of the index of the centroid with the minimum distance
            # traverse over the distances of the data point for each centroid and find the centroid with the minimum distance
            for j in range(1,k):    
                # if the distance of the data point to the current centroid is less than the minimum distance, then update the minimum distance
                if distances[i,j] < min_distance:
                    min_distance = distances[i,j]
                    cluster_index = j
                    # assign the data point to the centroid with the minimum distance
            cluster_assignments[i] = cluster_index

        # updating the centroids based on the new clustering assignments
        # traverse over each centroid and update it based on the data points assigned to it
        for i in range(k):
            # traverse over each dimension of the data points and update the centroid based on the data points assigned to it
            for j in range(data.shape[1]):
                # if there are no data points assigned to the centroid on the current dimension, then the centroid is set to 0
                if np.sum(cluster_assignments == i) == 0:
                    centroids[i,j] = 0
                # if there are data points assigned to the centroid on the current dimension, then the centroid is set to the mean of the data points assigned to it
                else:
                    centroids[i,j] = np.mean(data[cluster_assignments == i,j])
        iteration += 1
        break_loop = True
        # check for the loop breaking condition
        for i in range(data.shape[0]):
            if cluster_assignments[i] != prev_cluster_assignments[i]:
                break_loop = False
                break
    return centroids, cluster_assignments, iteration, distances

def silhouette_score_calculation(data, k, cluster_assignments, distances, centroids):
    silhouette_score = 0
    for i in range(len(data)):
        #for each element in the data, calculate the total distance of the element from all the other elements in the same cluster
        same_cluster_distance = 0
        count_same_cluster = 0
        for j in range(len(data)):
            if (cluster_assignments[i] == cluster_assignments[j]):
                same_cluster_distance += np.linalg.norm(data[i] - data[j])
                count_same_cluster += 1
        avg_same_cluster_distance = 0
        if (count_same_cluster != 0):
            avg_same_cluster_distance = same_cluster_distance / count_same_cluster
        #for each element in the data, calculate the total distance of the element from all the other elements of one cluster, and then do this for all the clusters
        #calculate the minimum of these distances
        min_other_cluster_distance = 0
        for j in range(k):
            if (cluster_assignments[i] != j):
                other_cluster_distance = 0
                count_other_cluster = 0
                for l in range(len(data)):
                    if (cluster_assignments[l] == j):
                        other_cluster_distance += np.linalg.norm(data[i] - data[l])
                        count_other_cluster += 1
                avg_other_cluster_distance = 0
                if (count_other_cluster != 0):
                    avg_other_cluster_distance = other_cluster_distance / count_other_cluster
                if (min_other_cluster_distance == 0):
                    min_other_cluster_distance = avg_other_cluster_distance
                else:
                    min_other_cluster_distance = min(min_other_cluster_distance, avg_other_cluster_distance)
        #calculate the silhouette score for the element
        silhouette_score_for_element = (min_other_cluster_distance - avg_same_cluster_distance) / max(avg_same_cluster_distance, min_other_cluster_distance)
        silhouette_score += silhouette_score_for_element
    return silhouette_score / len(data)        

# implementation of the fuzzy c-means algorithm
def fuzzy_c_means(data, no_of_clusters, m, beta, max_iterations, tolerance):
    # initialize the centroids

    centroids = np.zeros((no_of_clusters, data.shape[1]))
    centroids = np.random.rand(no_of_clusters, data.shape[1])

    # initialize the membership matrix
    # the membership matrix is used for storing the membership values of each data point to each cluster

    membership_matrix = np.zeros((data.shape[0], no_of_clusters))
    membership_matrix = np.random.rand(data.shape[0], no_of_clusters)
    
    # normalize the membership matrix
    # by normalizing the membership matrix, we ensure that the sum of the membership values of each data point to all clusters is equal to 1
    # the normalization is done by the formula:
    # membership_matrix[i, j] = membership_matrix[i, j] / sum(membership_matrix[i, :])
    # membership_matrix = membership_matrix / np.sum(membership_matrix, axis=1, keepdims=True)
    
    for i in range(membership_matrix.shape[0]):
        membership_matrix[i, :] = membership_matrix[i, :] / np.sum(membership_matrix[i, :])
    
    # initialize the distances matrix
    
    distances = np.zeros((data.shape[0], no_of_clusters))
    
    # initialize the iteration counter
    
    iteration = 0
    
    # initialize the change in centroids
    # the change in centroids is used for checking if the centroids have converged
    # if the change in centroids is less than the tolerance, the algorithm stops
    
    change_in_centroids = np.inf
    
    # loop until the maximum number of iterations is reached or the change in centroids is less than the tolerance, for tolerance almost zero
    
    while iteration < max_iterations and change_in_centroids > tolerance:
        # calculate the distances between the data points and the centroids
        # for i in range(no_of_clusters):
            # distances[:, i] = np.linalg.norm(data - centroids[i], axis=1)
    
        for i in range(no_of_clusters):
            for j in range(data.shape[0]):
                distances[j, i] = np.linalg.norm(data[j] - centroids[i])
    
        # update the membership matrix
    
        for i in range(membership_matrix.shape[0]):
            for j in range(no_of_clusters):
                membership_matrix[i, j] = 1 / (distances[i, j] ** (2 / (m - 1)))
    
        # normalize the membership matrix
        # membership_matrix = membership_matrix / np.sum(membership_matrix, axis=1, keepdims=True)
    
        for i in range(membership_matrix.shape[0]):
            membership_matrix[i, :] = membership_matrix[i, :] / np.sum(membership_matrix[i, :])
    
        # update the centroids
        # saving a copy of the centroids before updating them, this is a deep copy and not a shallow copy
    
        centroids_old = centroids.copy()
        for i in range(no_of_clusters):
            centroids[i] = np.sum(membership_matrix[:, i] ** m * data.T, axis=1) / np.sum(membership_matrix[:, i] ** m)
    
        # calculate the change in centroids
    
        change_in_centroids = np.linalg.norm(centroids - centroids_old)
    
        # increment the iteration counter
    
        iteration += 1
    # calculate the cluster assignments
    
    cluster_assignments = np.argmax(membership_matrix, axis=1)
    return centroids, cluster_assignments, distances, membership_matrix

# calculation of the objective function (J-value) for the fuzzy c-means algorithm
def J_value_calculation(data, no_of_clusters, m, beta, max_iterations, tolerance, centroids, cluster_assignments, distances, membership_matrix):
    # run the fuzzy c-means algorithm
    # calculate the J value
    J_value = 0
    for i in range(no_of_clusters):
        for j in range(data.shape[0]):
            J_value += membership_matrix[j, i] ** m * distances[j, i] ** 2
    return J_value