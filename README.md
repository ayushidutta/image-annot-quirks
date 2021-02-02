# Automatic Image Annotation: the quirks and what works

This repository contains the code for our paper published in Multimedia Tools and Applications: https://doi.org/10.1007/s11042-018-6247-3

- Non deep learning annotation models like 2PKNN, Tagprop, Tagrel, SVM, JEC
- Deep learning annotation models based on multi label loss functions like Softmax, Sigmoid, Pairwise Ranking, WARP, LSEP
- Empirical Experiments as per the paper
  - Per Label vs Per Image Evaluation Criteria 
  - Dataset Specific Biases
  
# Non-deep learning annotation models
Refer to setup.md

# Deep learning annotation models

Repo: https://github.com/ayushidutta/cnn-image-classification

# Empirical analysis

Refer to analysis.md

# Datasets

The 'data' folder contains the train/test splits of all datasets used in this experiment. For images, please refer to the individual dataset's page. 

# Requirements

- MATLAB
- Python 2
