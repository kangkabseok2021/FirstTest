# Auto Sampler
Repository contains a group of algorithms for conveniently and automatically sample diverse and balanced subset of a larger image dataset. This is useful for creating balanced a dataset before training a model. Since the algorithm does not rely on labels for clustering it can even be used on a unlabeled datasets. Thus it can be used to sample a diverse dataset already before labeling to save labeling effort.

# How it works
The main idea is to utilize image features from a pretrained neural network to group images that are similar clusters and then sampling uniformly accross clusters. As the clusters are used in feature space no images labels are needed.

The algorithm has 3 steps:

1. Compute image features for all images
2. Cluster images using their images features
3. Sample a subset images by sampling the most representative sample from each cluster

The algorithm uses and ml_core exported [CLIP](https://github.com/openai/CLIP) 'ViT-B/32' model for computing image features and [BIRCH](https://scikit-learn.org/stable/modules/clustering.html#birch) for clustering.

This 'ViT-B/32' model is converted to ONNX model and to *.tar which can be run by ml_core (by export_clip.py of pb_onnx_converter).

To use defaul model, you have to install ml-models or download "Clip_ViTB32.tar" in CLIP_auto_sampler to your current directory.

