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

```
rLabelSets(dData, fTrainAnnot, fTestAnnot)
```

##### [Sec 4.2.1 in paper, Table 8] Percentage “unique” and “novel” label-sets.

## Image Diversity

##### [Sec 4.2.2 in paper, Tables 9-12] Performance over the 20% most and least overlapping test subsets of various datasets.

