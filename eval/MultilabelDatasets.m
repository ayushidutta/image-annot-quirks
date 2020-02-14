classdef MultilabelDatasets
    %MUTLILABELDATASETS General properties about the dataset
    
    properties
    end
    
    methods (Static)
        
        function [labels_sort,labelFreq] = get_label_freq(annot,sortOrder)
            if (nargin<2)
                sortOrder=2;
            end;    
            n_labels = size(annot,2);
            labelFreq = zeros(n_labels,1);
            for i = 1:n_labels
                labelFreq(i) = sum(annot(:,i));
            end;
            labels_sort = [(1:n_labels)' labelFreq];
            labels_sort = sortrows(labels_sort,sortOrder);
            labels_sort = labels_sort(:,1);
        end    
           
        function [freq_perBin]=freqOverBins(binFreq,freq_sorted) 
            bin_sz = length(binFreq);
            n_val = length(freq_sorted);
            freq_perBin = zeros([bin_sz 1]);
            st_idx = 1;
            ed_idx = 1;
            for i=1:bin_sz
                while ed_idx <= n_val && freq_sorted(ed_idx)<=binFreq(i)
                    ed_idx = ed_idx +1;
                end;  
                if (st_idx<ed_idx)
                    freq_perBin(i) = ed_idx - st_idx; 
                end;  
                st_idx = ed_idx;
            end;    
        end
        
        function [uniq,freq_set,ia,ic] = get_labels_sets(annot)
            n_annot = size(annot,1);
            [uniq,ia,ic] = unique(annot,'rows','stable');
            cnt_uniq=size(uniq,1);
            uniqueness = cnt_uniq/n_annot*100.0;
            disp(['Unique is ' num2str(cnt_uniq) '/' num2str(n_annot) ' ' num2str(uniqueness) '%']);
            clear annot;
            freq_set=zeros(1,cnt_uniq);
            for i=1:cnt_uniq
                cnt_occur = find(ic==i);
                freq_set(i)=length(cnt_occur);    
            end;
            val=max(freq_set);
            idx = find(freq_set==val);
            disp(['Max occuring annotation:' num2str(val) 'times']);
            disp(num2str(idx));
            for i=1:length(idx)
                disp(['Row no:' num2str(idx(i))]);
                gt = find(uniq(idx(i),:));
                disp(num2str(gt));
            end;
        end
        
    end
    
end
