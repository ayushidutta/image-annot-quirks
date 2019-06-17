# Automatic Image Annotation: the quirks and what works

This repository contains the code for our paper published in Multimedia Tools and Applications: https://doi.org/10.1007/s11042-018-6247-3

Automatic image annotation is one of the fundamental problems in computer vision and machine learning. Given an image, here the goal is to predict a set of textual labels that describe the semantics of that image. During the last decade, a large number of image annotation techniques have been proposed that have been shown to achieve encouraging results on various annotation datasets. However, their scope has mostly remained restricted to quantitative results on the test data, thus ignoring various key aspects related to dataset properties and evaluation metrics that inherently affect the performance to a considerable extent. Here, first we evaluate ten state-of-the-art (both deep-learning based as well as non-deep-learning based) approaches for image annotation using the same baseline CNN features. Then we propose new quantitative measures to examine various issues/aspects in the image annotation domain, such as dataset specific biases, per-label versus per-image evaluation criteria, and the impact of changing the number and type of predicted labels. We believe the conclusions derived in this paper through thorough empirical analyzes would be helpful in making systematic advancements in this domain.

# Run the various annotation models and analysis

Refer to setup.md and analysis.md for details.

# Datasets

The 'data' folder contains the train/test splits of all datasets used in this experiment. For images, please refer to the individual dataset's page. 

# Requirements

- MATLAB
