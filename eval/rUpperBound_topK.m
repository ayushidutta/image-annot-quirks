function rUpperBound_topK( dData, ... % Data Directory
                fTrainAnnot, ... % Train Annotations file
                fTestAnnot, ... % Test Annotations File
                dScores, ... % Directory of label scores
                fScores, ... % Label scores file
                topK, ... % No of top labels to be assigned
                fillKMode ... % How to fill for remaining K[Rare,Frequent,Random]
               )
%REVAL Assigns correctly predicted labels to the image and fills rest with
%Rare labels

trainAnnot = load(fullfile(dData,fTrainAnnot));
testAnnot = load(fullfile(dData,fTestAnnot));
n_imgs = size(testAnnot,1);
testScores = load(fullfile(dScores,fScores));
testScores = testScores.testScores;
predictedAnnot1 = MultilabelAnnotate.annotateTopK(testScores',topK);
predictedAnnot1 = MultilabelAnnotate.annotateAintersectB(predictedAnnot1,testAnnot);

methd = {'Rare', 'Freq', 'Random'};
for fillKMode=1:3
    
predictedAnnot = MultilabelAnnotate.fillK(predictedAnnot1,trainAnnot,topK,fillKMode);

% Per Image
disp([methd{fillKMode} 'F1(L) / F1(I)']);
mv = MultilabelEvaluate(testAnnot,testScores',predictedAnnot);
resI = mv.calc_prec_rec_f1_map();
mv = MultilabelEvaluate(testAnnot',testScores,predictedAnnot');
resL = mv.calc_prec_rec_f1_map();   
disp(num2str([resL.f1 resI.f1]));

%mean_annot_I = sum(sum(predictedAnnot,2))/n_imgs;
%disp(['Mean labels assigned per image:' num2str(mean_annot_I)]);

end

end

