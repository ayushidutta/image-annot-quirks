function [ scores_ts,predict_ts ] = predict( dist_ts,Y_tr,K,w,topK )
%PREDICT 2PKNN Returns ranked labels scores 

n_test = size(dist_ts,1);
n_labels = size(Y_tr,2);
scores_ts = zeros(n_labels,n_test);
label_freq = sum(Y_tr);

tic;
for i=1:n_test

cnt_subset=0;   
subset_tr = zeros(1,n_labels*K);
% Pass 1
for j=1:n_labels 
    tr_idx = find(Y_tr(:,j)==1);
    curr_dist = dist_ts(i,tr_idx);
    for k=1:K
        if (k<=label_freq(j))
            cnt_subset=cnt_subset+1;   
            [~,idx] = min(curr_dist);
            subset_tr(cnt_subset)=tr_idx(idx);
            curr_dist(idx) = inf;
        else
            break;
        end;
    end;    
end;
subset_tr = subset_tr(1:cnt_subset);
subset_tr = unique(subset_tr); 
% Pass 2
curr_dist = dist_ts(i,subset_tr);
min_dist = min(curr_dist);
mean_dist = mean(curr_dist);
curr_dist = curr_dist-min_dist;
curr_dist = curr_dist/mean_dist;
curr_scores=exp(-w*curr_dist);
cnt_subset = length(subset_tr);
for j=1:cnt_subset
    tr_idx=subset_tr(j);
    tr_labels = find(Y_tr(tr_idx,:)==1);
    scores_ts(tr_labels,i)= scores_ts(tr_labels,i)+curr_scores(j);
end;    

scores_ts(:,i) = scores_ts(:,i)/sum(scores_ts(:,i));

end; % End n_test
toc;

if(nargin > 4)
    scores_ts=bsxfun (@rdivide, scores_ts, max(scores_ts,[],2));
    predict_ts = MultilabelAnnotate.annotateTopK(scores_ts',topK);
else
    predict_ts=[];
end;

end

