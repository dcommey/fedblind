# clustering/base_clustering.py

from abc import ABC, abstractmethod

class BaseClustering(ABC):
    def __init__(self, max_clusters=10, cluster_on='features'):
        self.max_clusters = max_clusters
        self.num_clusters = None
        self.model = None
        self.cluster_on = cluster_on

    @abstractmethod
    def cluster_clients(self, client_dataloaders):
        pass

    @abstractmethod
    def compute_client_statistics(self, client_dataloaders):
        pass

    @abstractmethod
    def find_optimal_clusters(self, data):
        pass

    @abstractmethod
    def compute_clustering_quality(self, data, labels):
        pass

class DPBaseClustering(BaseClustering):
    def __init__(self, max_clusters=10, epsilon=1.0, cluster_on='features'):
        super().__init__(max_clusters, cluster_on)
        self.epsilon = epsilon


class NonDPBaseClustering(BaseClustering):
    pass