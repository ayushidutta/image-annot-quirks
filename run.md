# Setup

Setup involves the following steps. 

### Prepare the Feature Files 
To run different annotation models, you need to first extract the features and save them in a mat file. The matfile containing the features has 2 fields: 
- _ftr_ : The features matrix (No. of features X Feature dimension)
- _n_ftrs_ : The no. of features. This corresponds to the no. of training and testing images for train and test features respectively.

To save large feature data, you may save the features across multiple feature files with file names indexed by \_1, \_2, and so on.

# Run different annotation models (2PKNN, SVM, Tagprop, Tagrel, JEC)

## Run 2PKNN
 
Initialize the 2PKNN class as specified below with the following arguments:
- _dData_ : Root data directory, where all the features, model and results directories are saved
- _dFtr_ : Feature directory where the feature files are present
- _dModel_ : Model directory where the 2PKNN models are present or saved
- _fTrainFtr_ : Training features file (.mat)
- _fTestFtr_ : Testing features file (.mat)
- _fTrainAnnot_ : Training annotations file (Refer to 'data' folder)
- _fTestAnnot_ : Testing annotations file (Refer to 'data' folder)
- _fModel_ : 2PKNN model file (.mat)
- _fResults_ : Results file where the 2PKNN predicted label scores will be saved (.mat)

 ```matlab
t = rTPKNN_v2('../data/nuswide','net-res1-101','tpknn','iapr_train_r101.mat','iapr_test_r101.mat','iapr_train_annot.txt','iapr_test_annot.txt','iapr_tpknn_r101_model.mat','iapr_test_r101_pred.mat');
```

Predict using 2PKNN.

 ```matlab
t.predict(-1,-1,4,1,3000,5000);
```
Evaluate 2PKNN performance.

```matlab
t.evalPerformance(3,1);
```
