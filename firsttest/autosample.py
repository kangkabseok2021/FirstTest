
import cv2
import fire
import numpy as np
import os
from tqdm import tqdm

from auto_sampler.auto_sampler import AutoSampleImageSet


def autosample(video_path, output_dir="output", images_per_batch=10, batch_size=1000,
               ml_core_model="Clip_ViTB32/"):

    print(f"Sampling {images_per_batch} frames every batch of {batch_size} frames")
    # Open the video file
    video = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)

    # Loop through the frames
    i = 0
    image_list = []

    print("Reading frames")
    ret = True
    while ret:
        # Read the current frame
        ret, frame = video.read()

        if ret:
            i += 1
            if i % batch_size != 0:
                # Convert BGR to RGB
                image_list.append(frame)
            else:
                batch_idx = i / batch_size
                print(f"Computing features for batch {batch_idx}")
                sampler = AutoSampleImageSet(model_path=ml_core_model)

                # Compute image features for batch
                image_feature_list = [sampler.compute_image_features(x) for x in tqdm(image_list)]
                image_features = np.concatenate(image_feature_list, axis=0)

                print("Clustering and sampling most representative images from batch")

                # Cluster images and select the most representative ones using their images
                # features
                most_representative_samples, cluster_dict = sampler.sample_representative_subset(
                    image_features, image_list, num_clusters=images_per_batch, k_per_cluster=1)
                print(len(most_representative_samples), len(image_list), images_per_batch)
                for idx, image in enumerate(most_representative_samples):

                    output_path = os.path.join(output_dir, f"frame_{batch_idx}_{idx:04d}.jpg")
                    print(f"Saving frame {output_path}")
                    cv2.imwrite(output_path, image)
                # reset list
                image_list = []


if __name__ == '__main__':
    fire.Fire(autosample)
