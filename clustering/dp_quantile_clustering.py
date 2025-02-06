# clustering/dp_quantile_clustering.py

import numpy as np
from scipy.stats import laplace
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from .base_clustering import DPBaseClustering
import logging

class DPQuantileClustering(DPBaseClustering):
    def __init__(self, max_clusters=10, epsilon=1.0, num_quantiles=5, min_quantiles=3, dataset_name='mnist', cluster_on='features'):
        super().__init__(max_clusters, epsilon, cluster_on)
        self.num_quantiles = max(num_quantiles, min_quantiles)
        self.dataset_name = dataset_name
        self.num_classes = self.get_num_classes(dataset_name)
        self.privacy_budget_spent = 0

    def get_num_classes(self, dataset_name):
        dataset_classes = {
            'mnist': 10,
            'cifar10': 10,
            'cifar100': 100,
            'svhn': 10,
            'har': 6  # HAR dataset has 6 classes
        }
        return dataset_classes.get(dataset_name, 10)  # Default to 10 if dataset is unknown

    def compute_private_quantile(self, data, q, epsilon):
        if epsilon <= 0:
            raise ValueError("Epsilon must be positive")
        
        n = len(data)
        sorted_data = np.sort(data)
        
        def utility_function(x):
            rank = np.searchsorted(sorted_data, x, side='right')
            return -abs(rank - q * n)
        
        utilities = np.array([utility_function(x) for x in sorted_data])
        
        probabilities = np.exp(epsilon * utilities / (2 * self.sensitivity()))
        probabilities_sum = np.sum(probabilities)
        
        if probabilities_sum == 0:
            logging.warning("All probabilities are zero. Returning median.")
            return np.median(data)
        
        probabilities /= probabilities_sum
        
        selected_index = np.random.choice(n, p=probabilities)
        self.privacy_budget_spent += epsilon
        return sorted_data[selected_index]

    def sensitivity(self):
        return 1  # The sensitivity of rank-based utility is 1

    def cluster_clients(self, client_dataloaders):
        client_stats = self.compute_client_statistics(client_dataloaders)
        
        if self.num_clusters is None:
            self.num_clusters = self.find_optimal_clusters(client_stats)
        
        self.model = KMeans(n_clusters=self.num_clusters, random_state=42)
        cluster_assignments = self.model.fit_predict(client_stats)
        
        quality_metrics = self.compute_clustering_quality(client_stats, cluster_assignments)
        
        return cluster_assignments, quality_metrics

    def compute_client_statistics(self, client_dataloaders):
        if self.cluster_on == 'features':
            return self.compute_feature_quantiles(client_dataloaders)
        elif self.cluster_on == 'labels':
            return self.compute_label_quantiles(client_dataloaders)
        else:
            raise ValueError("cluster_on must be either 'features' or 'labels'")

    def compute_feature_quantiles(self, client_dataloaders):
        client_stats = []
        quantiles = np.linspace(0, 1, self.num_quantiles + 2)[1:-1]
        
        # Determine number of features from the first batch of the first dataloader
        first_batch = next(iter(client_dataloaders[0]))[0]
        num_features = np.prod(first_batch.shape[1:])
        epsilon_per_quantile = self.epsilon / (self.num_quantiles * num_features)
        
        for dataloader in client_dataloaders:
            all_data = np.concatenate([batch.numpy() for batch, _ in dataloader], axis=0)
            flattened_data = all_data.reshape(all_data.shape[0], -1)
            
            quantile_features = np.array([
                [self.compute_private_quantile(flattened_data[:, feature], q, epsilon_per_quantile) 
                 for q in quantiles]
                for feature in range(flattened_data.shape[1])
            ]).flatten()
            
            client_stats.append(quantile_features)
        
        return np.array(client_stats)

    def compute_label_quantiles(self, client_dataloaders):
        client_stats = []
        quantiles = np.linspace(0, 1, self.num_quantiles + 2)[1:-1]
        epsilon_per_quantile = self.epsilon / self.num_quantiles
        
        for dataloader in client_dataloaders:
            all_labels = np.concatenate([labels.numpy() for _, labels in dataloader])
            
            label_dist = np.bincount(all_labels, minlength=self.num_classes) / len(all_labels)
            
            quantiles = [self.compute_private_quantile(label_dist, q, epsilon_per_quantile) for q in quantiles]
            
            client_stats.append(quantiles)
        return np.array(client_stats)

    def find_optimal_clusters(self, data):
        n_samples, n_features = data.shape
        max_clusters = min(self.max_clusters, n_samples - 1)
        
        if max_clusters < 2:
            return 1  # Not enough samples for meaningful clustering
        
        data_for_clustering = self.apply_pca_if_needed(data)
        
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(data_for_clustering)
            
            if len(np.unique(labels)) > 1:
                silhouette_scores.append(silhouette_score(data_for_clustering, labels))
            else:
                silhouette_scores.append(-1)  # Invalid score

        return np.argmax(silhouette_scores) + 2  # +2 because we started from k=2

    def apply_pca_if_needed(self, data):
        n_samples, n_features = data.shape
        if n_features > 50:  # You can adjust this threshold
            pca = PCA(n_components=min(50, n_samples - 1))
            return pca.fit_transform(data)
        return data

    def compute_clustering_quality(self, data, labels):
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
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

    def get_privacy_budget_spent(self):
        return self.privacy_budget_spent