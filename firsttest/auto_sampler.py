"""
This file modify auto_sampler.py with ml_core instead of out clip and torch.
We can convert any clip model to onnx model and *.tar for runninmg on ml_core
by using tools/export_onnx.py.
The model tar file can be found in ml-models repository
(https://gitlab.virtualbaker.com/MachineLearning/ml-models)
in 'ml-models/CLIP_auto_sampler/Clip_ViTB32.tar'
To construct AutoSampleImageSet(), you may give a path of model (*.tar).
The defaut path is 'ml-models/CLIP_auto_sampler/Clip_ViTB32.tar'
"""
import os

from collections import defaultdict
import warnings
import numpy as np
import scipy.stats
from sklearn.cluster import Birch
import cv2

from ml_core.base.runner import ModelRunner


class AutoSampleImageSet:
    def __init__(self, model_path=None):
        self.model = ModelRunner()
        if model_path is None:
            model_path = "ml-models/CLIP_auto_sampler/Clip_ViTB32.tar"
            warnings.warn("Use default Clip model to get image features at %s \n" % model_path)
        if not os.path.exists(model_path):
            raise Exception("Clip model does not ready. Istall ml-models or download the model.")
        else:
            self.model.load_model(model_path)

    @staticmethod
    def open_image(path):
        return cv2.imread(path)

    def compute_image_features(self, image):
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        input_name = self.model.get_input_variables_names()
        image_features = self.model.predict({input_name[0]: image})
        return image_features[0]["output"]

    def sample_representative_subset(
        self, image_features, image_list, num_clusters, k_per_cluster=1, sampling="default"
    ):
        """
        Sample a subset of images by clustering the feature vectors and sample the most
        representative k samples within each cluster.
        """

        # Cluster images
        birch = Birch(n_clusters=num_clusters, threshold=0.01).fit(image_features)
        cluster_ids = birch.predict(image_features)

        # Arange images in a dictionary where the key is the cluster id
        # and the value is a list of images belonging to that cluster
        cluster_dict = defaultdict(list)
        for cluster_id, file_name in zip(cluster_ids, image_list):
            cluster_dict[cluster_id].append(file_name)

        # Find most representative samples within each cluster
        if not isinstance(image_features, np.ndarray):
            image_features = np.array(image_features)
        most_representative_samples = []
        for k, v in cluster_dict.items():
            cluster_features = image_features[cluster_ids == k]

            if len(cluster_features) < k_per_cluster:
                # each cluster should have more than k_per_cluster images
                # raise ValueError("Some clusters have less than k images,
                # set k_per_cluster to a lower integer")
                warnings.warn(
                    "Some clusters have less than k images, "
                    + "all images from these clusters will be returned."
                    + " Set k_per_cluster to a lower integer to receive k images "
                    + " from each cluster.",
                    stacklevel=2,
                )
                most_representative_idx = self._find_most_representative_samples(
                    cluster_features, len(cluster_features), sampling
                )

            else:
                most_representative_idx = self._find_most_representative_samples(
                    cluster_features, k_per_cluster, sampling
                )

            for id in most_representative_idx:
                most_representative_samples.append(v[id])

        return most_representative_samples, cluster_dict

    def _find_most_representative_samples(self, image_features, k, sampling="default"):
        """
        Finds the most representative samples within a cluster of image features based
            on sampling method
        'default' sampling returns index of the closest sample to the center and
            k-1 random indices
        'central' sampling returns index of the k samples closest to the center
            of all samples in image_features

        """
        sampling_types = ["default", "central"]
        if sampling not in sampling_types:
            raise ValueError("Invalid sampling type. Expected one of: %s" % sampling_types)

        centroid = np.mean(image_features, axis=0)
        distances = np.linalg.norm(image_features - centroid, axis=1)

        if k == 1:
            return [np.argmin(distances)]

        elif sampling == "default":
            # sample k-1 random idx and insert id closest to center
            all_idx = np.arange(len(distances))
            id_closest = np.argmin(distances)
            all_idx = np.delete(all_idx, np.where(all_idx == id_closest))
            idx = np.random.choice(all_idx, size=k - 1, replace=False)
            idx = np.insert(idx, 0, id_closest)

        elif sampling == "central":
            # find idx with k smallest distances
            idx = np.argpartition(distances, k)[:k]

        return idx

    def plot_histogram(self, cluster_dict, fig_file=None):
        """
        Plot histogram of image clusters
        """

        entropy = self.entropy(cluster_dict)

        # visulaize histogram
        import matplotlib.pyplot as plt

        cluster_ids = self._cluster_ids(cluster_dict)
        plt.hist(cluster_ids, bins=np.max(cluster_ids) + 1)
        plt.title("entropy {:.3f}".format(entropy))
        plt.xlabel("bins")
        plt.ylabel("num images")
        if fig_file is not None:
            plt.savefig(fig_file)

    def entropy(self, cluster_dict):
        """
        Method computes entropy of clustered data
        This can be used as a metric of how balanced/inbalanced your data is.
        Entropy is between 0 and 1 where a higher value means a more balanced dataset
        (more uniformly distributed across feature space).
        """
        cluster_ids = self._cluster_ids(cluster_dict)
        hist = np.histogram(cluster_ids, bins=np.max(cluster_ids) + 1)
        entropy = scipy.stats.entropy(hist[0], base=len(hist[0]))
        return entropy

    def _cluster_ids(self, cluster_dict):
        """Convert cluster dict to a list with cluster ids"""
        cluster_ids = []
        for k, v in cluster_dict.items():
            cluster_ids.extend([k] * len(v))
        return cluster_ids
