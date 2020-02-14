function rEvalPartition_topK( dData, ... % Data Directory
                fTestAnnot, ... % Test Annotations File
                dModel, ... % Directory model
                ftrModel, ... Ftr Model
                dScores, ... % Directory of label scores
                fScores, ... % List of label scores file
                topK ... % No of top labels to be assigned
               )
%REVAL Evaluates performance of a method and bars

testAnnot = load(fullfile(dData,fTestAnnot));
testScores = load(fullfile(dScores,fScores));
testScores = testScores.testScores;
predictedAnnot = MultilabelAnnotate.annotateTopK(testScores',topK);
matModel = matfile(fullfile(dModel,ftrModel));
novelIdx=matModel.novelIdx_ts;
not_novelIdx=matModel.notNovelIdx_ts;
            
% Novel Test Images **************
disp('Novel images');

% Per Image
mv = MultilabelEvaluate(testAnnot(novelIdx,:),testScores(:,novelIdx)',predictedAnnot(novelIdx,:));
resI = mv.calc_prec_rec_f1_map();
disp('Performance per image(Prec/Rec/F1/N+/MAP) :');
disp(num2str([resI.prec resI.rec resI.f1 resI.nplus resI.map]));

% Per Label
mv = MultilabelEvaluate(testAnnot(novelIdx,:)',testScores(:,novelIdx),predictedAnnot(novelIdx,:)');
resL = mv.calc_prec_rec_f1_map();   
disp('Performance per label(Prec/Rec/F1/N+/MAP) :');
disp(num2str([resL.prec resL.rec resL.f1 resL.nplus resL.map]));

% Not Novel Test Images **************
disp('Not Novel images');

% Per Image
mv = MultilabelEvaluate(testAnnot(not_novelIdx,:),testScores(:,not_novelIdx)',predictedAnnot(not_novelIdx,:));
resI = mv.calc_prec_rec_f1_map();
disp('Performance per image(Prec/Rec/F1/N+/MAP) :');
disp(num2str([resI.prec resI.rec resI.f1 resI.nplus resI.map]));

% Per Label
mv = MultilabelEvaluate(testAnnot(not_novelIdx,:)',testScores(:,not_novelIdx),predictedAnnot(not_novelIdx,:)');
resL = mv.calc_prec_rec_f1_map();   
disp('Performance per label(Prec/Rec/F1/N+/MAP) :');
disp(num2str([resL.prec resL.rec resL.f1 resL.nplus resL.map]));


end

