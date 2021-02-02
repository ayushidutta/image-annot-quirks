function rGroundUB_topK( dData, ... % Data Directory
                fTrainAnnot, ... % Train Annotations file
                fTestAnnot, ... % Test Annotations File
                topK, ... % No of top labels to be assigned
                fillKMode ... % How to fill for remaining K[Rare,Frequent,Random]
               )
%REVAL Evaluates performance of a method and bars

trainAnnot = load(fullfile(dData,fTrainAnnot));
testAnnot = load(fullfile(dData,fTestAnnot));
testScores = testAnnot'; % DUMMY

for fillKMode=1:3
    predictedAnnot = MultilabelAnnotate.fillK(testAnnot,trainAnnot,topK,fillKMode);
    % Per Image & Label
    mv = MultilabelEvaluate(testAnnot,testScores',predictedAnnot);
    resI = mv.calc_prec_rec_f1_map();
    mv = MultilabelEvaluate(testAnnot',testScores,predictedAnnot');
    resL = mv.calc_prec_rec_f1_map();   
    disp('F1(L) / F1(I) :');
    disp(num2str([resL.f1 resI.f1]));
end

% Random Assignments 
%%{
disp('Simply assigning rare / freq / random labels !!');
for i=1:3
    predictedAnnot = zeros(size(testAnnot));
    predictedAnnot = MultilabelAnnotate.fillK(predictedAnnot,trainAnnot,topK,i);
    mv = MultilabelEvaluate(testAnnot',testScores,predictedAnnot');
    resL = mv.calc_prec_rec_f1_map();   
    disp('Performance per label(Prec/Rec/F1/N+) :');
    disp(num2str([resL.prec resL.rec resL.f1 resL.nplus]));
    mv = MultilabelEvaluate(testAnnot,testScores',predictedAnnot);
    resI = mv.calc_prec_rec_f1_map();
    disp('Performance per image(Prec/Rec/F1/N+) :');
    disp(num2str([resI.prec resI.rec resI.f1 resI.nplus]));
end;
%%}

end

