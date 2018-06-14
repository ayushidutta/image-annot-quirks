classdef rSVM
    %RSVM SVM Classification
    % run('/home/ayushi/lib/vlfeat-0.9.20/toolbox/vl_setup');
    
    methods(Static)
        
        function train(fTrainFtr,fTrainAnnot,fModel,C,B0)
            matFtrs = matfile(fTrainFtr);
            tr_ftrs = matFtrs.ftr;
            tr_labels = load(fTrainAnnot);
            [n_train,n_labels] = size(tr_labels);
            ftr_norm=sqrt(sum(tr_ftrs.^2, 2));
            tr_ftrs=bsxfun (@rdivide, tr_ftrs, ftr_norm);
            ftr_dim = size(tr_ftrs,2);
            W = zeros(ftr_dim,n_labels);
            B = zeros(n_labels,1);
            sigA = zeros(n_labels,1);
            sigB = zeros(n_labels,1);
            lambda = 1/(C*n_train);
            for i=1:n_labels  
                if (mod(i,40)==0)
                    disp(['Label:' num2str(i)]); 
                end;    
                class1 = find(tr_labels(:,i)==1);
                Y_tr = zeros(n_train,1) - 1;
                Y_tr(class1) = 1;
                prior1 = length(class1);
                prior0 = n_train-prior1;
%               class0 = tr_labels(:,i)~=1;                
%               cw = zeros(1,n_train);
%               cw(class1)=n_train/prior1;
%               cw(class0)=n_train/prior0; %,'Weights',cw - Not better 
                [W(:,i),B(i),~,tr_scores] = vl_svmtrain(tr_ftrs',Y_tr,lambda,'BiasMultiplier',B0, ...
                      'MaxNumIterations',n_train*100); % 'Solver','SGD','Loss','L2',
                [sigA(i),sigB(i)] = sigmoidPlatt(tr_scores,Y_tr',prior0,prior1);
            end
            matModel=matfile(fModel,'Writable',true);
            matModel.W=W;
            matModel.B=B;
            matModel.sigA=sigA;
            matModel.sigB=sigB;
            matModel.C=C;
            matModel.B0=B0;
        end
        
        % Using Lib Linear 
            % dir_lib = '/home/ayushi/Repo/liblinear/liblinear-2.1/matlab';addpath(dir_lib);
        function trainTest_liblin(fTrainFtr,fTrainAnnot,fTestFtr,fTestAnnot,C,topK)
            matFtrs = matfile(fTrainFtr);
            tr_ftrs = matFtrs.ftr;
            tr_labels = load(fTrainAnnot);
            [n_train,n_labels] = size(tr_labels);
            Y_ts = load(fTestAnnot);
            ftr_norm=sqrt(sum(tr_ftrs.^2, 2));
            tr_ftrs=bsxfun (@rdivide, tr_ftrs, ftr_norm);
            matFtrs = matfile(fTestFtr);
            ts_ftrs = matFtrs.ftr;
            ftr_norm=sqrt(sum(ts_ftrs.^2, 2));
            ts_ftrs=bsxfun (@rdivide, ts_ftrs, ftr_norm);
            ftr_dim = size(tr_ftrs,2);
            W = zeros(ftr_dim,n_labels);
            B = zeros(n_labels,1);
            n_test=size(ts_ftrs,1);
            scores_ts=zeros(n_labels,n_test);
            for i=1:n_labels  
                if (mod(i,40)==0)
                    disp(['Label:' num2str(i)]); 
                end;    
                class1 = tr_labels(:,i)==1;
                Y_tr = zeros(n_train,1) - 1;
                Y_tr(class1) = 1;
                opt = ['-c' num2str(C) '-s 2 -B 1'];
                model = train(Y_tr,sparse(tr_ftrs), opt);
                W(:,i) = model.w(1,1:ftr_dim)';
                B(i) = model.bias;
                [~, ~, dec_values] = predict(randn(n_test,1), sparse(ts_ftrs), model);
                scores_ts(i,:) = dec_values(:,1)';
            end
            predict_ts = MultilabelAnnotate.annotateTopK(scores_ts',topK);
            mv = MultilabelEvaluate(Y_ts',scores_ts,predict_ts');
            resL = mv.calc_prec_rec_f1_map();  
            disp('Performance per label(Prec/Rec/F1/N+/MAP) :');
            disp(num2str([resL.prec resL.rec resL.f1 resL.nplus resL.map]));
            mv = MultilabelEvaluate(Y_ts,scores_ts',predict_ts);
            resI = mv.calc_prec_rec_f1_map();  
            disp('Performance per image(Prec/Rec/F1/N+/MAP) :');
            disp(num2str([resI.prec resI.rec resI.f1 resI.nplus resI.map]));
        end
        
        function predict(fTestFtr,fModel,fScores,C,B0)
            matModel = matfile(fModel);
            matFtrs = matfile(fTestFtr);
            ts_ftrs = matFtrs.ftr;
            ftr_norm=sqrt(sum(ts_ftrs.^2, 2));
            ts_ftrs=bsxfun (@rdivide, ts_ftrs, ftr_norm);
            n_test=size(ts_ftrs,1);
            W=matModel.W;
            B=matModel.B;
            sigA=matModel.sigA;
            sigB=matModel.sigB;
            n_labels=length(B);
            ts_prob = zeros(n_labels,n_test);
            ts_scores = ts_ftrs*W;
            ts_scores = ts_scores';
            for i = 1:n_labels  % TODO 
                ts_scores(i,:) = ts_scores(i,:) + B(i);
            end;
            for i = 1:n_labels  % TODO 
                for j = 1:n_test
                    ts = ts_scores(i,j);
                    prob = 1/(1+exp(sigA(i)*ts+sigB(i)));
                    ts_prob(i,j) = prob;
                end;
            end;
            matScores=matfile(fScores,'Writable',true);
            matScores.testScores=ts_prob;
        end
        
        function [resL] = tuneParamCV(fTrainFtr,fTrainAnnot,ftrModel,C,B0,fold,topK)
            matFtrs = matfile(fTrainFtr);
            tr_ftrs = matFtrs.ftr;
            tr_labels = load(fTrainAnnot);
            matModel = matfile(ftrModel);
            cvSplit=matModel.cvSplit;
            trainIdx = cvSplit.trainIdx(fold,:);
            testIdx = cvSplit.testIdx(fold,:);
            model=MultilabelSVM.train(tr_ftrs(trainIdx,:),tr_labels(trainIdx,:),C,B0);
            scores_ts=MultilabelSVM.predict(tr_ftrs(testIdx,:),model);
            predict_ts = MultilabelAnnotate.annotateTopK(scores_ts',topK);
            mv = MultilabelEvaluate(tr_labels(testIdx,:)',scores_ts,predict_ts');
            resL = mv.calc_prec_rec_f1_map(); 
            disp(['F1(L)=' num2str(resL.f1)]);
        end
        
        function selectBestLabelPerf(fTrainFtr,fTrainAnnot,ftrModel,fModel,C,B0,topK)
            Y_tr=load(fTrainAnnot);
            [~,n_labels]=size(Y_tr);
            clear Y_tr;
            model = matfile(ftrModel);
            cvSplit=model.cvSplit;
            kfold = length(cvSplit.TrainSize);
            bestF1_label = zeros(kfold,n_labels);
            for fold=1:kfold
                disp(['Fold: ' num2str(fold)]);
                resL=rSVM.tuneParamCV(fTrainFtr,fTrainAnnot,ftrModel,C,B0,fold,topK);
                bestF1_label(fold,:) = resL.f1_row';
            end;
            model = matfile(fModel,'Writable',true);
            model.bestF1_label = mean(bestF1_label);             
        end
        
    end
    
end

