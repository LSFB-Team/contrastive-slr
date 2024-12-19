import torch
from sklearn.neighbors import KNeighborsClassifier

from cslr.metrics import ClassificationFullMetrics
from cslr.metrics.utils import log_metrics_to_dir
from cslr.config import KNNConfig


def evaluate_knn_classifier(
        instance_ids: dict[str, list[str]],
        embeddings_path: str,
        log_dir: str,
        knn_config: KNNConfig,
):
    print("-- loading embeddings")
    embeddings_data = torch.load(embeddings_path, weights_only=True, map_location='cpu')

    print("-- training KNN classifier...")
    train_ids = set(instance_ids['training'])
    train_indices = [idx for idx, instance_id in enumerate(embeddings_data['ids']) if instance_id in train_ids]
    train_embeddings = embeddings_data['embeddings'][train_indices]
    train_labels = embeddings_data['labels'][train_indices]

    knn_classifier = KNeighborsClassifier(n_neighbors=knn_config.n_neighbors, metric=knn_config.metric)
    knn_classifier = knn_classifier.fit(train_embeddings, train_labels)

    print("-- computing predictions...")
    val_ids = set(instance_ids['validation'])
    val_indices = [idx for idx, instance_id in enumerate(embeddings_data['ids']) if instance_id in val_ids]
    val_embeddings = embeddings_data['embeddings'][val_indices]
    val_labels = embeddings_data['labels'][val_indices]

    probs = torch.from_numpy(knn_classifier.predict_proba(val_embeddings))

    print("-- computing and saving metrics...")
    metric_module = ClassificationFullMetrics(prefix="knn/", n_classes=500)
    metrics = metric_module(probs, val_labels)
    log_metrics_to_dir(log_dir, metrics)
