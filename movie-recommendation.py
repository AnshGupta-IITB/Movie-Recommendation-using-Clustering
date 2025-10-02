# Movie Recommendation System using Clustering (K-Means)
# Dataset: MovieLens 100k

# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load Dataset
column_names = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('ratings.csv')

movies = pd.read_csv('movies.csv', on_bad_lines='warn')
movies = movies[[0, 1]]
movies.columns = ['movie_id', 'title']

# Merge ratings with movie titles
data = pd.merge(ratings, movies, on='movie_id')

# Step 3: Create User-Movie Matrix
user_movie_matrix = data.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

# Step 4: Feature Scaling
scaler = StandardScaler()
user_movie_scaled = scaler.fit_transform(user_movie_matrix)

# Step 5: Determine Optimal Number of Clusters
sil_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(user_movie_scaled)
    sil_scores.append(silhouette_score(user_movie_scaled, kmeans.labels_))

plt.figure(figsize=(8,5))
plt.plot(range(2, 11), sil_scores, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Optimal Number of Clusters')
plt.show()

# Step 6: Fit K-Means with chosen clusters (example: k=4)
kmeans = KMeans(n_clusters=4, random_state=42)
user_clusters = kmeans.fit_predict(user_movie_scaled)
user_movie_matrix['cluster'] = user_clusters

# Step 7: Movie Recommendation Function
def recommend_movies(user_id, num_recommendations=5):
    cluster = user_movie_matrix.loc[user_id, 'cluster']
    cluster_users = user_movie_matrix[user_movie_matrix['cluster'] == cluster]
    cluster_mean = cluster_users.drop('cluster', axis=1).mean()
    user_rated = user_movie_matrix.loc[user_id].drop('cluster')
    recommendations = cluster_mean[user_rated == 0].sort_values(ascending=False).head(num_recommendations)
    return recommendations.index.tolist()

# Example Recommendation
print("Movies recommended for User 1:")
print(recommend_movies(1))

# Step 8: Visualize Clusters
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(user_movie_scaled)
plt.figure(figsize=(8,6))
plt.scatter(reduced_data[:,0], reduced_data[:,1], c=user_clusters, cmap='viridis', alpha=0.6)
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('User Clusters based on Movie Ratings')
plt.show()
