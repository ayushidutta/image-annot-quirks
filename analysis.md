# Analysis

Refer to our MTA paper (https://doi.org/10.1007/s11042-018-6247-3) for details regarding specific experiments.
 
## Per-label versus Per-image Evaluation

##### [Sec 4.1 in paper, Tables 5-6] Performance by replacing incorrect predictions with rare/frequent/random incorrect labels.
##### [Sec 4.1 in paper, Table 7] Performance by assigning the three most rare, the three most frequent, and three randomly chosen labels to each test image. 

```
rUpperBound_topK(dData, fTrainAnnot, fTestAnnot, dScores, fScores, topK, fillKMode)  

rGroundUB_topK(dData, fTrainAnnot, fTestAnnot, topK, fillKMode)  
```
Args: 
- _dData_: Data directory
- _fTrainAnnot_: Train Annotations file
- _fTestAnnot_: Test Annotations File
- _dScores_: Directory of label scores
- _fScores_: Label scores file
- _topK_: No of top labels to be assigned
- _fillKMode_: How to fill for remaining K, choices 1, 2 or 3 for Rare, Frequent, Random.

E.g:
```
rUpperBound_topK('','coco_train_annot.txt','coco_test_annot.txt','net-incep3','tpknn/coco_test_g3K_pred.mat',3,0) 

rGroundUB_topK('','coco_train_annot.txt','coco_test_annot.txt',3,1)  
```

## Label Diversity

##### [Sec 4.2.1 in paper, Table 8] Percentage “unique” and “novel” label-sets

```
rLabelSets(dData, fTrainAnnot, fTestAnnot)
```
Refer to above for arguments description.

## Image Diversity

##### [Sec 4.2.2 in paper, Tables 9-12] Performance over the 20% most and 20% least overlapping test subsets of various datasets

Find the list of the 20% most and least overlapping test image indices and save it in the model file.
```
rPartitionList.rankByDist(dModel, ftrModel, dTestDist, fTestDist, K, batch)
```
Args:
- _dModel_: Model directory
- _ftrModel_: Model filename
- _dTestDist_: Directory containing the precomputed test-train distance file
- _fTestDist_: Precomputed test-train distance file
- _K_: No of Nearest Neighbours
- _batch_: Batch size i.e. how many to compute at a time

Evaluate the performances of a list of methods on the above subsets.
```
rCmpMethds_visualSim_v2(dData, fTestAnnot, dScores, fScoresList, methdNames, dModel, ftrModel, p, topK)
```
Args:
- _fScoresList_: List of label score files 
- _methdNames_: List of method names
- _p_: Partition Top and bottom percentage e.g. 0.20

(Refer to above for remaining arguments)

E.g.
```
rPartitionList.rankByDist('models', 'esp_gnetKCCA_model.mat','distances','esp_test_gnetKCCA_L2',5,3000);

rCmpMethds_visualSim_v2('','esp_test_annot.txt','net-res1-101',...
    {'tpknn/esp_test_r101K_pred.mat','tagprop/esp_test_r101K_pred.mat',...
    'svm/esp_test_r101K_pred.mat','tagrel/esp_test_r101K_pred.mat','jec/esp_test_r101K_pred.mat'}, ...
    {'2PKNN+KCCA','Tagprop+KCCA','SVM+KCCA','Tagrel+KCCA','Jec+KCCA'},'net-res1-101','esp_train_r101K_model.mat',0.20,5); 
```

