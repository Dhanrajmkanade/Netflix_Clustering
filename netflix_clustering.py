import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('netflix.csv')
print("Initial data sample:")
print(df.head())

# Clean 'duration' column: extract digits, convert to numeric, fill missing with 0, convert to int
df['duration'] = pd.to_numeric(df['duration'].str.extract('(\d+)')[0], errors='coerce')
df['duration'] = df['duration'].fillna(0).astype(int)

# Encode categorical columns 'listed_in' (genres) and 'rating'
le = LabelEncoder()
for col in ['listed_in', 'rating']:
    df[col] = le.fit_transform(df[col].astype(str))

# Select features for clustering
features = df[['listed_in', 'rating', 'duration']]

# Initialize and fit KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(features)

print("\nCluster assignment sample:")
print(df[['title', 'listed_in', 'rating', 'duration', 'Cluster']].head())

# Visualize clustering result
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='duration', y='rating', hue='Cluster', palette='viridis')
plt.title('Netflix Shows Clustering')
plt.xlabel('Duration (minutes)')
plt.ylabel('Rating (encoded)')
plt.legend(title='Cluster')
plt.show()

