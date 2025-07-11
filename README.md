# Netflix Show Clustering

This project performs clustering analysis on a Netflix dataset to group shows based on various features like genre, duration, country, and rating.

## Project Overview

- Load and clean the Netflix dataset.
- Preprocess categorical features using Label Encoding.
- Convert duration to numeric values.
- Apply KMeans clustering to group similar shows.
- Visualize and analyze the clusters.

## Dataset

The dataset should be named `netflix.csv` and placed in the project root folder. It contains columns such as:

- show_id
- type
- title
- director
- cast
- country
- date_added
- release_year
- rating
- duration
- listed_in (genres)
- description

## Requirements

Install the required Python packages using:

pip install -r requirements.txt

Author
Dhanraj Kanade
Contact: dkanade988@gmail.com