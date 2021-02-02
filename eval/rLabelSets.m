function rLabelSets( dData, ... % Data Directory
                fTrainAnnot, ... % Train Annotations file
                fTestAnnot ... % Test Annotations file               
                )
%RLABELSETS Unique and novel label sets

trainAnnot = load(fullfile(dData,fTrainAnnot));
n_train=size(trainAnnot,1);
testAnnot = load(fullfile(dData,fTestAnnot));
n_test=size(testAnnot,1);

% Unique Train label set
[uniq_tr,~,ic] = unique(trainAnnot,'rows','stable');
cnt_tr=size(uniq_tr,1);
uniqueness = cnt_tr/n_train*100.0;
disp(['Train Unique is ' num2str(cnt_tr) '/' num2str(n_train) ' ' num2str(uniqueness) '%']);
clear trainAnnot;
freq_set=zeros(1,cnt_tr);
for i=1:cnt_tr
    cnt_occur = find(ic==i);
    freq_set(i)=length(cnt_occur);    
end;
val=max(freq_set);
idx = find(freq_set==val);
disp(['Max occuring annotation:' num2str(val) 'times']);
disp(num2str(idx));
for i=1:length(idx)
    disp(['Row no:' num2str(idx(i))]);
    gt = find(uniq_tr(idx(i),:));
    disp(num2str(gt));
end;    

%%{
% Unique Test label set
[uniq_ts,~,ic] = unique(testAnnot,'rows','stable');
cnt_ts=size(uniq_ts,1);
uniqueness = cnt_ts/n_test*100.0;
disp(['Test Unique is ' num2str(cnt_ts) '/' num2str(n_test) ' ' num2str(uniqueness) '%']);
freq_set=zeros(1,cnt_ts);
for i=1:cnt_ts
    cnt_occur = find(ic==i);
    freq_set(i)=length(cnt_occur);    
end;
val=max(freq_set);
idx = find(freq_set==val);
disp(['Max occuring annotation:' num2str(val) 'times']);
disp(num2str(idx));
for i=1:length(idx)
    disp(['Row no:' num2str(idx(i))]);
    gt = find(uniq_ts(idx(i),:));
    disp(num2str(gt));
end   
disp('Novel check');
cnt_novel=0;
for i=1:n_test 
    if (mod(i,1000)==0)
        disp(num2str(i));
    end;  
    match=false;
    for j=1:cnt_tr
        label_set = find(testAnnot(i,:)-uniq_tr(j,:));
        if (~any(label_set))
            match=true;
            break;
        end;    
    end;
    if(~match)
        cnt_novel=cnt_novel+1;
    end;    
end;
novelness = cnt_novel/n_test * 100.0;
disp(['Novel is ' num2str(cnt_novel) '/' num2str(n_test) ' ' num2str(novelness) '%']);
%%}

end

