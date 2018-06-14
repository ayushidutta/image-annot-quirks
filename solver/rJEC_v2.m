classdef rJEC_v2
    %RJEC Run JEC
    
    properties
        fTrainAnnot
        fTestAnnot  
        fModel
        fTrainFtr
        fTestFtr
        fScores
        isTest
    end
    
    methods
        
        function [obj] = rJEC_v2(dData,dModel,dResults,fTrainFtr,fTestFtr,fTrainAnnot,fTestAnnot ,fModel,fResults)
            obj.fTrainAnnot = fullfile(dData, fTrainAnnot);
            obj.fTestAnnot = fullfile(dData, fTestAnnot);
            dModel = fullfile(dData,dModel);            
            obj.fModel = fullfile(dModel, fModel);
            obj.fTrainFtr = fullfile(dModel,fTrainFtr);
            obj.fTestFtr = fullfile(dModel,fTestFtr);
            dResults = fullfile(dModel,dResults);  
            obj.fScores = fullfile(dResults,fResults);   
            if strcmpi(fTrainFtr,fTestFtr)
                obj.isTest = false;
            else
                obj.isTest = true;
            end
            disp(obj.isTest);
        end
        
        function predict(obj,test_split_idx, test_split_sz, topK, batch1, batch2)
            fFtr1 = matfile(obj.fTestFtr);
            fFtr2 = matfile(obj.fTrainFtr);
            ftrDist = FtrDist(fFtr1, fFtr2);
            n_ftrs = fFtr1.n_ftrs;
            if test_split_idx > 0
                st_ftr = (test_split_idx-1) * test_split_sz + 1;
                [path,fname,ext] = fileparts(obj.fScores);
                matScores = matfile(fullfile(path,[fname '_' num2str(test_split_idx) ext]),'Writable',true); 
                ed_ftr = min(n_ftrs, st_ftr+test_split_sz-1);
            else
                matScores = matfile(obj.fScores,'Writable',true);
                st_ftr=1;
                ed_ftr = n_ftrs;
            end    
            matModel = matfile(obj.fModel);
            maxDist=matModel.maxDist;            
            n_rows=ed_ftr-st_ftr+1; 
            Y_tr = sparse(load(obj.fTrainAnnot));
            n_labels=size(Y_tr,2);
            scores_ts=zeros(n_labels,n_rows);
            for i=1:batch1:n_rows
                batch1 = min(batch1,n_rows-i+1);
                st_idx = st_ftr+i-1;
                dist_ts = ftrDist.calc_dist(st_idx, batch1, batch2);    
                dist_ts = dist_ts/maxDist;
                if ~obj.isTest
                    tr_idxs=(1:batch1) + st_idx -1;
                    dist_ts(1:batch1,tr_idxs)=inf;                    
                end    
                scores_ts(:,i:i+batch1-1)=jec(dist_ts',Y_tr,topK);
            end;
            clear dist_ts;
            matScores.testScores = scores_ts;
            matScores.st_row=st_ftr;
            matScores.ed_row=ed_ftr;        
        end
        
        function evalPerformance(obj,topK,n_tsFiles)
            Y_ts=sparse(load(obj.fTestAnnot));
            [n_test,n_labels]=size(Y_ts);
            scores_ts=zeros(n_labels,n_test);
            [path,fname,ext] = fileparts(obj.fScores);
            for i=1:n_tsFiles  
                if n_tsFiles>1
                    matScores = matfile(fullfile(path,[fname '_' num2str(i) ext]));                 
                else
                    matScores = matfile(obj.fScores);
                end   
                st_row=matScores.st_row;
                ed_row=matScores.ed_row;
                scores_ts(:,st_row:ed_row)= matScores.testScores;
            end;
            matScores = matfile(obj.fScores,'Writable',true);
            matScores.testScores = scores_ts;
            predict_ts = MultilabelAnnotate.annotateTopK(scores_ts',topK);
            mv = MultilabelEvaluate(Y_ts',scores_ts,predict_ts');
            resL = mv.calc_prec_rec_f1_map();  
            disp('Performance per label(Prec/Rec/F1/N+/MAP) :');
            disp(num2str([resL.prec resL.rec resL.f1 resL.nplus resL.map]));
            mv = MultilabelEvaluate(Y_ts,scores_ts',predict_ts);
            resI = mv.calc_prec_rec_f1_map();  
            disp('Performance per image(Prec/Rec/F1/N+/MAP) :');
            disp(num2str([resI.prec resI.rec resI.f1 resI.nplus resI.map]));
        end;
        
    end
    
end

