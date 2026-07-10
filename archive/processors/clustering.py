import pandas as pd
from sklearn.cluster import KMeans

def cluster_events(events):
    X = events[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=5)
    clusters = kmeans.fit_predict(X)

    events['cluster'] = clusters
    return events
