classdef MultilabelSVM

    methods (Static)
        
        function [model]=train(tr_ftrs,tr_labels,C,B0)
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
                [W(:,i),B(i),~,tr_scores] = vl_svmtrain(tr_ftrs',Y_tr,lambda,'BiasMultiplier',B0, ...
                      'MaxNumIterations',n_train*100); % 'Solver','SGD','Loss','L2',
                [sigA(i),sigB(i)] = sigmoidPlatt(tr_scores,Y_tr',prior0,prior1);
            end
            model.W=W;
            model.B=B;
            model.sigA=sigA;
            model.sigB=sigB;
        end
       
        function [ts_prob]=predict(ts_ftrs,model)
            ftr_norm=sqrt(sum(ts_ftrs.^2, 2));
            ts_ftrs=bsxfun (@rdivide, ts_ftrs, ftr_norm);
            n_test=size(ts_ftrs,1);
            W=model.W;
            B=model.B;
            sigA=model.sigA;
            sigB=model.sigB;
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
        end
        
    end
    
end

