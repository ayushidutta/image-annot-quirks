# Setup

Setup involves the following steps. 

### Prepare the Feature Files 
To run different annotation models, you need to first extract the features and save them in a mat file. The matfile containing the features has 2 fields: 
- _ftr_ : The features matrix (No. of features X Feature dimension)
- _n_ftrs_ : The no. of features. This corresponds to the no. of training and testing images for train and test features respectively.

To save large feature data, you may save the features across multiple feature files with file names indexed by \_1, \_2, and so on. If _N_ is the the total no. of features split across _n_ files, each file has _N_/_n_ no. features except the last file which can have the remaining features.

# Run different annotation models (2PKNN, SVM, Tagprop, Tagrel, JEC)

## Run 2PKNN
 
Initialize the 2PKNN class as specified below with the following arguments:
- _dData_ : Root data directory, where all the features, annotations, model and results directories and files are saved
- _dFtr_ : Feature directory where the feature files are present
- _dModel_ : Model directory where the 2PKNN models are present or saved
- _fTrainFtr_ : Training features file (.mat)
- _fTestFtr_ : Testing features file (.mat)
- _fTrainAnnot_ : Training annotations file (Refer to 'data' folder)
- _fTestAnnot_ : Testing annotations file (Refer to 'data' folder)
- _fModel_ : 2PKNN model file (.mat)
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
## Run Tagprop

