# Prepare the dataset

To run different annotation models, you need to first extract the features and save them in a mat file. 

### Feature File
The matfile containing the features has 2 fields: 
- _ftr_ : The features matrix (No. of features X Feature dimension)
- _n_ftrs_ : The no. of features. This corresponds to the no. of training and testing images for train and test features respectively.

To save large feature data, you may save features across multiple feature files.

# Run different annotation models (2PKNN, SVM, Tagprop, Tagrel, JEC)

## Run 2PKNN
 
 Initialize the 2PKNN class.
-----------------------------
Specify the following arguments fir rTPKNN_v2:
- 

 ```matlab
t = rTPKNN_v2('../data/nuswide','net-res1-101','tpknn','iapr_train_r101.mat','iapr_test_r101.mat','iapr_train_annot.txt','iapr_test_annot.txt','iapr_train_r101_model.mat','iapr_test_r101_pred.mat');
t.predict(-1,-1,4,1,3000,5000);
t.evalPerformance(3,1);
```
Predict using 2PKNN.
---------------------

 ```matlab
t = rTPKNN_v2('../data/nuswide','net-res1-101','tpknn','iapr_train_r101.mat','iapr_test_r101.mat','iapr_train_annot.txt','iapr_test_annot.txt','iapr_train_r101_model.mat','iapr_test_r101_pred.mat');
t.predict(-1,-1,4,1,3000,5000);
t.evalPerformance(3,1);
```

