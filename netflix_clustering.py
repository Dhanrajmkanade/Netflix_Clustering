import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('netflix.csv')
print("Initial data sample:")
print(df.head())

# Clean and preprocess the 'duration' column
# Extract numeric values from strings (e.g., "90 min" -> 90)
df['duration'] = df['duration'].str.extract('(\d+)')  # Extract digits as strings
df['duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0).astype(int)

# Encode categorical columns: 'listed_in' (genres) and 'rating'
label_encoders = {}
for col in ['listed_in', 'rating']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Store encoder for possible inverse transform later

# Select features for clustering
features = df[['listed_in', 'rating', 'duration']]

# Initialize and fit the KMeans clustering model
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(features)

print("\nSample cluster assignments:")
print(df[['title', 'listed_in', 'rating', 'duration', 'cluster']].head())

# Visualize clusters: duration vs rating colored by cluster
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='duration', y='rating', hue='cluster', palette='viridis', alpha=0.7)
plt.title('Netflix Shows Clustering')
plt.xlabel('Duration (minutes)')
plt.ylabel('Rating (encoded)')
plt.legend(title='Cluster')
plt.tight_layout()
plt.show()


