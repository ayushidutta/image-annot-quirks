classdef rPartitionList
    %RRANKLIST Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
    end
    
    methods (Static)
        
        function rankByDist(  dModel, ... % Directory model
                              ftrModel, ... Ftr Model
                              dTestDist, ... % Test Dist directory
                              fTestDist, ... % File Test Dist directory
                              K, ... No of Nearest Neighbours
                              batch ... Batch Size 
                             )       
            matModel=matfile(fullfile(dModel,ftrModel),'Writable',true);
            n_tsFiles = matModel.n_tsFiles;
            n_test = matModel.n_test;
            test_sortBySim = zeros(n_test,2);
            test_sortBySim(:,1)= (1:n_test)';
            maxDist=matModel.maxDist;
            for i=1:n_tsFiles
                curr_f=matfile(fullfile(dTestDist,[fTestDist '_' num2str(i) '.mat']));
                n_rows=curr_f.n_rows;
                st_row = curr_f.st_row;
                batch_j = batch;
                for j=1:batch_j:n_rows
                    batch_j = min(batch_j,n_rows-j+1);
                    dist_ts=curr_f.dist(j:j+batch_j-1,:);
                    dist_ts = dist_ts/maxDist;
                    st_idx=st_row+j-1;
                    ed_idx=st_idx+batch_j-1;
                    disp(['NN:' num2str(st_idx) '-' num2str(ed_idx)]);
                    [~,ND]=getNN(dist_ts,batch_j,K,0);
                    test_sortBySim(st_idx:ed_idx,2)=mean(ND)';
                end;    
            end;
            clear dist_ts;
            test_sortBySim = sortrows(test_sortBySim,2);
            matModel.test_byL2Sim=test_sortBySim(:,1);
        end
        
        
        function byNovelLabelSets( dData, ... % Data Directory
                                   fTrainAnnot, ... % Train Annotations file
                                   fTestAnnot, ... % Test Annotations file
                                   dModel, ... % Directory model
                                   ftrModel ... Ftr Model
                                 )
            trainAnnot = load(fullfile(dData,fTrainAnnot));
            testAnnot = load(fullfile(dData,fTestAnnot));
            n_test = size(testAnnot,1);
            matModel = matfile(fullfile(dModel,ftrModel),'Writable',true);
            uniq_tr = unique(trainAnnot,'rows','stable');
            cnt_tr=size(uniq_tr,1);
            uniq_ts = unique(testAnnot,'rows','stable');
            cnt_ts=size(uniq_ts,1);
            uniqueness = cnt_ts/n_test*100.0;
            disp(['Test Unique is ' num2str(cnt_ts) '/' num2str(n_test) ' ' num2str(uniqueness) '%']);
            cnt_novel=0;
            novelIdx=[];
            not_novelIdx=[];
            for i=1:n_test 
                if (mod(i,1000)==0)%{
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
                    novelIdx = [novelIdx i];
                else
                    not_novelIdx = [not_novelIdx i];
                end;    
            end;
            novelness = cnt_novel/n_test * 100.0;
            disp(['Novel is ' num2str(cnt_novel) '/' num2str(n_test) ' ' num2str(novelness) '%']);
            matModel.novelIdx_ts=novelIdx;
            matModel.notNovelIdx_ts=not_novelIdx;
        end
        
    end
    
end

