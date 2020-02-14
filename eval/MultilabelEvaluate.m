classdef MultilabelEvaluate < handle
    %MULTILABELEVALUATE Performance Evaluation for Multilabel assignments,
    %calculated on a per row basis, i.e. N(images) * L(labels) 
    
    properties
        groundAnnot
        predictedAnnot         
        results
    end
    
    properties (Access = private)
        labelRank
        nCorrect_row
        nGround_row
        nPredict_row
        n_rows
        n_cols
    end
    
    methods
        
        function [obj] = MultilabelEvaluate(groundAnnot,labelScores,predictedAnnot)
            obj.groundAnnot = groundAnnot;
            obj.predictedAnnot = predictedAnnot;
            obj.n_rows = size(groundAnnot,1);
            obj.n_cols = size(groundAnnot,2);
            obj.labelRank = zeros([obj.n_rows obj.n_cols]);
            obj.nCorrect_row = zeros([obj.n_rows 1]);
            obj.nGround_row = zeros([obj.n_rows 1]);
            obj.nPredict_row = zeros([obj.n_rows 1]);
            for i=1:obj.n_rows
                ground = find(obj.groundAnnot(i,:)==1);
                obj.nGround_row(i) = length(ground);
                predicted = find(obj.predictedAnnot(i,:)==1);
                obj.nPredict_row(i) = length(predicted);
                for j=ground
                    if (obj.predictedAnnot(i,j)==1)
                        obj.nCorrect_row(i) = obj.nCorrect_row(i)+1;
                    end;    
                end;  
                labelScores_sort = [(1:obj.n_cols)' labelScores(i,:)'];
                labelScores_sort = sortrows(labelScores_sort,-2);
                obj.labelRank(i,:) = labelScores_sort(:,1)';
            end; 
        end
               
        % Evaluate all Performance Measures
        function [results] = calc_prec_rec_f1_map(obj)
            obj.calc_precision();
            obj.calc_recall();
            obj.calc_f1();
            obj.calc_map();
            results = obj.results;
        end    
            
        % Evaluate Precision
        function [prec,prec_row] = calc_precision(obj)
            prec_row = zeros([obj.n_rows 1]);
            for i=1:obj.n_rows
                if(obj.nPredict_row(i)>0)
                    prec_row(i) = obj.nCorrect_row(i)/obj.nPredict_row(i);
                end;    
            end;    
            prec = mean(prec_row);
            obj.results.prec = prec;
            obj.results.prec_row = prec_row;            
        end
        
        % Evaluate Recall
        function [rec,nplus,rec_row] = calc_recall(obj)
            rec_row = zeros([obj.n_rows 1]);
            for i=1:obj.n_rows
                if(obj.nGround_row(i)>0)
                    rec_row(i) = obj.nCorrect_row(i)/obj.nGround_row(i);
                end;    
            end;  
            posRec = find(rec_row);
            rec = mean(rec_row);
            nplus = length(posRec);
            obj.results.rec = rec;
            obj.results.rec_row = rec_row;
            obj.results.nplus = nplus;
        end
        
        % Evaluate F1
        function [f1,f1_row] = calc_f1(obj)
            if (~isfield(obj.results,'prec'))
                obj.calc_precision();
            end;    
            if (~isfield(obj.results,'rec'))
                obj.calc_recall();
            end;
            f1=0;
            f1_row = zeros([obj.n_rows 1]);
            for i=1:obj.n_rows
                s = obj.results.prec_row(i)+obj.results.rec_row(i);
                if(s>0)
                    f1_row(i) = (2*obj.results.prec_row(i)*obj.results.rec_row(i))/s;
                end;    
            end; 
            s = obj.results.prec+obj.results.rec;
            if (s>0)
                f1 = (2 * obj.results.prec * obj.results.rec) / s;
            end;
            obj.results.f1 = f1;
            obj.results.f1_row = f1_row;
        end
        
        % Evaluate Mean Average Precision
        function [map,ap_row] = calc_map(obj)
            ap_row = zeros([obj.n_rows 1]);
            for i=1:obj.n_rows
                nc=0;
                ap=0;
                for j=1:obj.n_cols
                    if (obj.groundAnnot(i,obj.labelRank(i,j))==1)
                        nc = nc+1;
                        ap = ap + nc/j;
                    end;    
                end;
                if (obj.nGround_row(i)>0)
                    ap_row(i) = ap/obj.nGround_row(i);
                end;    
            end;    
            map = mean(ap_row);
            obj.results.map = map;
            obj.results.ap_row = ap_row;
        end
          
        % Evaluate Break Even Point Precision
        function [bep,bep_row] = calc_bep(obj)
            bep_row = zeros([obj.n_rows 1]);
            for i=1:obj.n_rows
                nc=0;
                for j=1:obj.nGround_row(i)
                    if (obj.groundAnnot(i,obj.labelRank(j))==1)
                        nc = nc+1;
                    end;    
                end;
                bep_row(i) = nc/obj.nGround_row(i);
            end;    
            bep = mean(bep_row);
            obj.results.bep = bep;
            obj.results.bep_row = bep_row;
        end
        
    end
    
    methods (Static)
        
        function [mean_perBin]=avgOverBinFreq(binFreq,freq_sorted,val_freqSorted) 
            bin_sz = length(binFreq);
            n_val = length(val_freqSorted);
            mean_perBin = zeros([bin_sz 1]);
            st_idx = 1;
            ed_idx = 1;
            for i=1:bin_sz
                while ed_idx <= n_val && freq_sorted(ed_idx)<=binFreq(i)
                    ed_idx = ed_idx +1;
                end;  
                if (st_idx<ed_idx)
                    mean_perBin(i) = mean(val_freqSorted(st_idx:ed_idx-1));
                end;  
                st_idx = ed_idx;
            end;    
        end
        
    end
end

