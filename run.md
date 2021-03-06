## Code and Data Setup

Setup involves the following steps. 

### Code Path

Make sure that this 'image-annot-quirks' folder and its subfolders containing all code, are added to the MATLAB path in order to execute each function in MATLAB. 

### Prepare the Feature Files 
To run different annotation models, you need to first extract the features and save them in a mat file. The matfile containing the features has 2 fields: 
- _ftr_ : The features matrix (No. of features X Feature dimension)
- _n_ftrs_ : The no. of features. This corresponds to the no. of training and testing images for train and test features respectively.

To save large feature data, you may save the features across multiple feature files with file names indexed by \_1, \_2, and so on. If _N_ is the the total no. of features split across _n_ files, each file has _N_/_n_ no. features except the last file which can have the remaining features.

### Set up features distance related statistics

Nearest neighbour models are based on the distance of the test samples from the training samples. In order to normalise the distances, we calculated the maximum possible distance prior to running our models. We also store certain other statistics such as the K nearest neighbour indices for the test samples, etc. that may needed for our analysis later. This can be done by running the following function.

 ```matlab
model_dist( fFtr1,fFtr2,fModel,fNN,batch1,batch2)
 ```
where,
- _fFtr1_ : Testing features file (.mat)
- _fFtr2_ : Training features file (.mat)
- _fModel_ : Model file where the results will be saved (.mat)
- _fNN_ : File where the nearest neghbour indices will be saved (.mat)
- _batch1_ : Integer specifying the no. of test samples in memory at a time, ideally 5000. This is decided based on feature dimension, and affects only the speed of the code.
- _batch2_ : Integer specifying the no. of train samples in memory at a time, ideally 5000. This is decided based on feature dimension, and affects only the speed of the code.

## Non deep learning annotation models (2PKNN, SVM, Tagprop, Tagrel, JEC)

### Run 2PKNN
 
Initialize the 2PKNN class as specified below with the following arguments:
- _dData_ : Root data directory, where all the features, annotations, model and results directories and files are saved
- _dFtr_ : Feature directory where the feature files are present
- _dModel_ : Model directory where the 2PKNN models are present or saved
- _fTrainFtr_ : Training features file (.mat)
- _fTestFtr_ : Testing features file (.mat)
- _fTrainAnnot_ : Training annotations file (Refer to 'data' folder)
- _fTestAnnot_ : Testing annotations file (Refer to 'data' folder)
- _fModel_ : 2PKNN model file (.mat). Create this file by copying the feature model file created in the setup section.
- _fResults_ : Results file where the 2PKNN predicted label scores will be saved (.mat)

 ```matlab
t = rTPKNN_v2('data/nuswide','net-res1-101','tpknn','nus1_train_r101.mat','nus1_test_r101.mat','nus1_train_annot.txt','nus1_test_annot.txt','nus1_tpknn_r101_model.mat','nus1_test_r101_pred.mat');
```

Predict using 2PKNN. This saves the 2PKNN scores in _fResults_ file. 
Args:
- _test_split_idx_ : In case of multiple test feature files, specify the file index 1, 2 and so on, else default to -1
- _test_split_sz_ : In case of multiple test feature files, no. of features in each file else default to -1.
- K : The hyperparameter _K_
- w : The hyperparameter _w_
- batch1 : Integer specifying the no. of test samples in memory at a time. This is decided based on feature dimension, and affects only the speed of the code.
- batch2 : Integer specifying the no. of train samples in memory at a time. This is decided based on feature dimension, and affects only the speed of the code.

 ```matlab
t.predict(-1,-1,4,1,3000,5000);
```
Evaluate 2PKNN performance. This displays the per-label and per-image performance scores (F1, Prec, Rec, MAP).
Args:
- _topK_ : The no. of top labels assigned as true predictions from the label scores. 
- _n_tsFiles_ : In case of multiple test feature files, the no. of test feature files, else default to -1.

```matlab
t.evalPerformance(3,1);
```

### Run Tagprop

Follow the original paper on 'Tagprop' for installation of the library. The code has been referred here in the 'lib/tagProp' folder.

To run the training and testing follow the following steps. Check the arguments as explained above for 2PKNN. 

```matlab
t = rTAGPROP_v2(dData, dFtr, dModel, fTrainFtr, fTestFtr, fTrainAnnot, fTestAnnot, fModel, fResults);
t.train(K, batch1, batch2);
t.predict(test_split_idx, test_split_sz, K, batch1, batch2);
t.evalPerformance(topK, n_tsFiles);
```
where _K_ is a hyperparameter for Tagprop.

### Run Tagrel

Similar to 2PKNN arguments above, run the following:

```matlab
t = rTAGREL_v2(dData, dFtr, dModel, fTrainFtr, fTestFtr, fTrainAnnot, fTestAnnot, fModel, fResults);
t.predict(test_split_idx, test_split_sz, K, batch1, batch2);
t.evalPerformance(K, topK, n_tsFiles);
```
where _K_ is a hyperparameter for Tagrel.

### Run JEC

Similar to 2PKNN arguments above, run the following:

```matlab
t = rJEC_v2(dData, dFtr, dModel, fTrainFtr, fTestFtr, fTrainAnnot, fTestAnnot, fModel, fResults);
t.predict(test_split_idx, test_split_sz, topK, batch1, batch2);
t.evalPerformance(topK, n_tsFiles);
```

### Run SVM

Install the 'VLFEAT' library for SVM. Run the following code for training and testing. 

```matlab
rSVM.train(fTrainFtr,fTrainAnnot,fModel,C,B0);
rSVM.predict(fTestFtr,fModel,fScores);
```
where, _C_, _B0_ are the SVM hyperparameters and rest follow the arguments as mentioned for 2PKNN above.

## KCCA Feature extraction

Similar to 2PKNN arguments, train KCCA and compute the train features.
```
t = rKCCA_v2(dData,dModel,fTrainFtr,fTestFtr,fTrainAnnot,fModel,fTrainKcca,fTestKcca)
t.train(batch1,batch2)
```
_fTrainKcca_ and _fTestKcca_ refer to filenames where KCCA train and test features will get saved. Once the training is complete, before computing the test KCCA features, set the feature distance related statistics, as shown above, calling the function _model_dist()_. With the new model file, compute test features as below.

```
t = rKCCA_v2(dData,dModel,fTrainFtr,fTestFtr,fTrainAnnot,fModel,fTrainKcca,fTestKcca)
t.project(batch1,batch2)
```
