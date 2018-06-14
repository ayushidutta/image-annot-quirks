classdef TagRelevance
    %TAGRELEVANCE 
    
    methods (Static)
        
        function [scores_ts]= predict(NN,Y_tr,K)            
            n_test = size(NN,2);
            [n_train,n_labels] = size(Y_tr);
            label_freq = sum(Y_tr);
            scores_ts = zeros(n_labels,n_test);
            label_prob = label_freq/n_train;
            label_prob = label_prob';
            for i = 1:n_test    
                kt = zeros(n_labels,1);
                for j = 1:K
                    tags = find(Y_tr(NN(j,i),:)==1);
                    n_tags = length(tags);
                    for k=1:n_tags
                        kt(tags(k)) = kt(tags(k))+1;
                    end;    
                end;    
                scores_ts(:,i) = kt - K*label_prob;
            end;
        end
        
    end
    
end

