# clustering/non_dp_quantile_clustering.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from .base_clustering import NonDPBaseClustering
import logging
import torch
from typing import List, Any
from torch.utils.data import DataLoader

class NonDPQuantileClustering(NonDPBaseClustering):
    def __init__(self, max_clusters=10, num_quantiles=5, dataset_name='mnist', cluster_on='features'):
        super().__init__(max_clusters, cluster_on)
        self.num_quantiles = num_quantiles
        self.dataset_name = dataset_name
        self.num_classes = self.get_num_classes(dataset_name)

    def get_num_classes(self, dataset_name):
        dataset_classes = {
            'mnist': 10,
            'cifar10': 10,
            'cifar100': 100,
            'svhn': 10,
            'har': 6  # HAR dataset has 6 classes
        }
        return dataset_classes.get(dataset_name, 10)  # Default to 10 if dataset is unknown

    def compute_client_statistics(self, clients: List[Any]) -> np.ndarray:
        """Compute statistics for each client's data."""
        return self.compute_feature_quantiles(clients)

    def compute_feature_quantiles(self, clients: List[Any]) -> np.ndarray:
        """
        Compute quantiles of features for each client's dataset.
        
        Args:
            clients: List of client objects, each with a dataloader property
            
        Returns:
            np.ndarray: Array of shape (num_clients, num_features * num_quantiles)
                       containing quantile statistics for each client
        """
        all_client_stats = []
        
        for client in clients:
            dataloader = client.get_dataloader(batch_size=32)  # Use client's method to get dataloader
            features = []
            
            # Collect all features from this client's data
            for batch, _ in dataloader:
                if isinstance(batch, torch.Tensor):
                    features.append(batch.view(batch.size(0), -1).numpy())
                else:
                    features.append(np.array(batch).reshape(len(batch), -1))
            
            if features:
                features = np.concatenate(features, axis=0)
                # Compute quantiles for each feature
                quantiles = np.percentile(
                    features,
                    np.linspace(0, 100, self.num_quantiles),
                    axis=0
                )
                # Flatten the quantiles array
                client_stats = quantiles.flatten()
                all_client_stats.append(client_stats)
        
        return np.array(all_client_stats)

    def compute_label_quantiles(self, client_dataloaders):
        client_stats = []
        for dataloader in client_dataloaders:
            all_labels = []
            for _, labels in dataloader:
                all_labels.extend(labels.numpy())
            
            # Compute label distribution
            label_dist = np.bincount(all_labels, minlength=self.num_classes) / len(all_labels)
            
            # Compute quantiles of the label distribution
            quantiles = np.quantile(label_dist, np.linspace(0, 1, self.num_quantiles + 2)[1:-1])
            
            client_stats.append(quantiles)
        return np.array(client_stats)

    def cluster_clients(self, clients: List[Any], num_clusters=2) -> List[int]:
        """
        Cluster clients based on their data distribution.
        
        Args:
            clients: List of client objects
            num_clusters: Number of clusters to create
            
        Returns:
            List[int]: Cluster assignments for each client
        """
        if not clients:
            return []
            
        # Compute statistics for each client
        client_stats = self.compute_client_statistics(clients)
        
        if len(client_stats) < num_clusters:
            # If we have fewer clients than clusters, adjust num_clusters
            num_clusters = len(client_stats)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        cluster_assignments = kmeans.fit_predict(client_stats)
        
        return cluster_assignments.tolist()

    def find_optimal_clusters(self, data):
        n_samples, n_features = data.shape
        max_clusters = min(self.max_clusters, n_samples - 1)
        
        if max_clusters < 2:
            return 1  # Not enough samples for meaningful clustering
        
        # If the data is high-dimensional, apply PCA
        if n_features > 50:  # You can adjust this threshold
            pca = PCA(n_components=min(50, n_samples - 1))
            data = pca.fit_transform(data)
        
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data)
            
            if k < n_samples:
                silhouette_scores.append(silhouette_score(data, labels))
            else:
                silhouette_scores.append(-1)  # Invalid score

        return np.argmax(silhouette_scores) + 2  # +2 because we started from k=2

    def compute_clustering_quality(self, data, labels):
        if len(np.unique(labels)) < 2:
            return {
                "silhouette_score": None,
                "calinski_harabasz_score": None,
                "davies_bouldin_score": None
            }
        
        return {
            "silhouette_score": silhouette_score(data, labels),
            "calinski_harabasz_score": calinski_harabasz_score(data, labels),
            "davies_bouldin_score": davies_bouldin_score(data, labels)
        }

    def get_cluster_clients(self, clients: List[Any], cluster_id: int, cluster_assignments: List[int]) -> List[Any]:
        """
        Get all clients belonging to a specific cluster.
        
        Args:
            clients: List of all client objects
            cluster_id: ID of the cluster to get clients for
            cluster_assignments: List of cluster assignments for each client
            
        Returns:
            List[Any]: List of clients belonging to the specified cluster
        """
        return [client for client, assignment in zip(clients, cluster_assignments) 
                if assignment == cluster_id]