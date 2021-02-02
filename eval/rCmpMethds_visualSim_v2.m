function rCmpMethds_visualSim_v2(dData, ... % Data Directory
                              fTestAnnot, ... % Test Annotations File
                              dScores, ... % Directory of label scores
                              fScoresList, ... % label scores MAT file list
                              methdNames, ... % Method names
                              dModel, ... % Directory model
                              ftrModel, ... Ftr Model
                              p, ... % Partition Top and bottom index
                              topK ... % No of top labels to be assigned
                             )       
Y_ts = sparse(load(fullfile(dData,fTestAnnot))); 
matModel=matfile(fullfile(dModel,ftrModel));
test_sortBySim = matModel.test_byL2Sim;
n_test = size(Y_ts,1);
p = ceil(p*n_test);
topIdx = test_sortBySim(1:p);
bottomIdx = test_sortBySim(n_test-p+1:n_test);
n_methds = length(fScoresList);
topIdx_labelFreq = sum(Y_ts(topIdx,:));
bottomIdx_labelFreq = sum(Y_ts(bottomIdx,:));
topIdx_labels = find(topIdx_labelFreq>0);
bottomIdx_labels = find(bottomIdx_labelFreq>0);

% Performance of every method - Top Idx
disp(['Evaluating for top ' num2str(p) 'images']);
for i=1:n_methds
    disp(['Method:' methdNames{i} ' ******']);
    testScores = load(fullfile(dScores,fScoresList{i}));
    testScores = testScores.testScores;
    topIdxScores = testScores(topIdx_labels,topIdx);
    predictedAnnot = MultilabelAnnotate.annotateTopK(topIdxScores',topK);
    mv = MultilabelEvaluate(Y_ts(topIdx,topIdx_labels)',topIdxScores,predictedAnnot');
    resL = mv.calc_prec_rec_f1_map();   
    disp('Performance per label(Prec/Rec/F1/N+/MAP) :');
    disp(num2str([resL.prec resL.rec resL.f1 resL.nplus resL.map])); 
    mv = MultilabelEvaluate(Y_ts(topIdx,topIdx_labels),topIdxScores',predictedAnnot);
    resI = mv.calc_prec_rec_f1_map();   
    disp('Performance per image(Prec/Rec/F1/N+/MAP) :');
    disp(num2str([resI.prec resI.rec resI.f1 resI.nplus resI.map])); 
end;

% Performance of every method - Bottom Idx
disp(['Evaluating for bottom ' num2str(p) 'images']);
for i=1:n_methds
    disp(['Method:' methdNames{i} ' ******']);
    testScores = load(fullfile(dScores,fScoresList{i}));
    testScores = testScores.testScores;
    bottomIdxScores = testScores(bottomIdx_labels,bottomIdx);
    predictedAnnot = MultilabelAnnotate.annotateTopK(bottomIdxScores',topK);
    mv = MultilabelEvaluate(Y_ts(bottomIdx,bottomIdx_labels)',bottomIdxScores,predictedAnnot');
    resL = mv.calc_prec_rec_f1_map();   
    disp('Performance per label(Prec/Rec/F1/N+/MAP) :');
    disp(num2str([resL.prec resL.rec resL.f1 resL.nplus resL.map])); 
    mv = MultilabelEvaluate(Y_ts(bottomIdx,bottomIdx_labels),bottomIdxScores',predictedAnnot);
    resI = mv.calc_prec_rec_f1_map();   
    disp('Performance per image(Prec/Rec/F1/N+/MAP) :');
    disp(num2str([resI.prec resI.rec resI.f1 resI.nplus resI.map])); 
end;

%{

n_bars = ceil(n_test/p);
precL_bar = zeros(n_bars,n_methds);
recL_bar = zeros(n_bars,n_methds);
f1L_bar = zeros(n_bars,n_methds);

for i=1:n_methds
    testScores = load(fullfile(dScores,fScoresList{i}));
    testScores = testScores.testScores;    
    batch = p;
    cnt_bar=0;
    for j=1:batch:n_test
        batch = min(batch,n_test-j+1);
        cnt_bar = cnt_bar+1;
        testIdx = test_sortBySim(j:j+batch-1);
        testIdxScores=testScores(:,testIdx);
        predictedAnnot = MultilabelAnnotate.annotateTopK(testIdxScores',topK);
        mv = MultilabelEvaluate(Y_ts(testIdx,:)',testIdxScores,predictedAnnot');
        resL = mv.calc_prec_rec_f1_map();   
        disp('Performance per label(Prec/Rec/F1/N+/MAP) :');
        disp(num2str([resL.prec resL.rec resL.f1 resL.nplus resL.map])); 
        precL_bar(cnt_bar,i)=resL.prec;
        recL_bar(cnt_bar,i)=resL.rec;
        f1L_bar(cnt_bar,i)=resL.f1;
    end;
end;

% Plot Precision Bin wise 
figure('units','normalized','outerposition',[0 0 1 1])
bar(precL_bar)
set(gca,'XTick',1:n_bars);
title('Precision per overlapping images')
xlabel('Sorted overlapping images')
ylabel('Precision')
leg=legend(methdNames);
pause(1)
set(leg,'Location','BestOutside')

% Plot Recall Bin wise 
figure('units','normalized','outerposition',[0 0 1 1])
bar(recL_bar)
set(gca,'XTick',1:n_bars);
title('Recall per overlapping images')
xlabel('Sorted overlapping images')
ylabel('Recall')
leg=legend(methdNames);
pause(1)
set(leg,'Location','BestOutside')

% Plot F1 Bin wise 
figure('units','normalized','outerposition',[0 0 1 1])
bar(f1L_bar)
set(gca,'XTick',1:n_bars);
title('F1 per overlapping images')
xlabel('Sorted overlapping images')
ylabel('F1')
leg=legend(methdNames);
pause(1)
set(leg,'Location','BestOutside')

%}

end

