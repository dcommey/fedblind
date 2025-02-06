import tensorflow as tf
from federated.base_server import BaseServer
from federated.base_client import BaseClient
from clustering.kmeans_clustering import KMeansClustering
from knowledge_distillation.distiller import Distiller
from privacy.differential_privacy import DifferentialPrivacy

class FedSiKDServer(BaseServer):
    def __init__(self, config):
        super().__init__(config)
        self.clustering = KMeansClustering(config.num_clusters) if config.use_clustering else None
        self.privacy = DifferentialPrivacy(config) if config.use_differential_privacy else None

    def train_round(self, communication_round):
        if self.clustering:
            cluster_assignments = self.clustering.cluster_clients(self.clients)
        else:
            cluster_assignments = [0] * len(self.clients)

        global_weights = self.global_model.get_weights()
        client_weights = []

        for cluster in range(max(cluster_assignments) + 1):
            cluster_clients = [client for client, assignment in zip(self.clients, cluster_assignments) if assignment == cluster]
            
            # Train teacher (first client in the cluster)
            teacher_weights = cluster_clients[0].train(global_weights, is_teacher=True)
            
            # Train students (rest of the clients in the cluster)
            student_weights_list = [client.train(teacher_weights, is_teacher=False) for client in cluster_clients[1:]]
            
            # Aggregate weights within the cluster
            cluster_weights = self.aggregate_weights([teacher_weights] + student_weights_list)
            client_weights.append(cluster_weights)

        # Apply differential privacy if enabled
        if self.privacy:
            client_weights = [self.privacy.apply(weights) for weights in client_weights]

        # Aggregate weights across clusters
        new_weights = self.aggregate_weights(client_weights)
        self.global_model.set_weights(new_weights)

class FedSiKDClient(BaseClient):
    def __init__(self, client_id, data, labels, config):
        super().__init__(client_id, data, labels, config)
        self.distiller = Distiller(config) if config.use_knowledge_distillation else None

    def train(self, global_weights, is_teacher=False):
        if is_teacher or not self.distiller:
            self.model.set_weights(global_weights)
            self.model.fit(self.data, self.labels, epochs=self.config.local_epochs, verbose=0)
        else:
            student_weights = self.distiller.distill(self, global_weights)
            self.model.set_weights(student_weights)

        return self.model.get_weights()

class FedSiKD:
    def __init__(self, config):
        self.config = config
        self.server = FedSiKDServer(config)

    def run(self, clients_data):
        self.server.initialize_clients(clients_data)

        for round in range(self.config.num_rounds):
            self.server.train_round(round)

            if round % self.config.evaluate_every == 0:
                loss, accuracy = self.server.evaluate_global_model(self.config.test_data, self.config.test_labels)
                print(f"Round {round}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        return self.server.global_model

    def get_privacy_params(self):
        if self.server.privacy:
            return {
                'epsilon': self.server.privacy.get_epsilon(),
                'delta': self.server.privacy.get_delta()
            }
        return None

    def get_client_features(self):
        if self.server.clustering:
            return self.server.clustering.get_client_features()
        return None

    def get_cluster_assignments(self):
        if self.server.clustering:
            return self.server.clustering.get_cluster_assignments()
        return None