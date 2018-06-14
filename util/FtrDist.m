classdef FtrDist
    %FTRDIST Calculates feature distances
    
    properties
        mat_ftr1
        mat_ftr2
        n_ftrs1
        n_ftrs2
    end
    
    methods
        
        function [obj] = FtrDist(mat_ftr1,mat_ftr2)
           obj.mat_ftr1=mat_ftr1;
           obj.mat_ftr2=mat_ftr2;
           obj.n_ftrs1 = obj.mat_ftr1.n_ftrs;
           obj.n_ftrs2 = obj.mat_ftr2.n_ftrs;           
        end
       
        function [ftr_dist] = calc_dist(obj, st_ftr1, batch1, batch2)
            ftr1=double(obj.mat_ftr1.ftr(st_ftr1:st_ftr1+batch1-1,:));
            ftr1_norm=sqrt(sum(ftr1.^2, 2));
            ftr1=bsxfun (@rdivide, ftr1, ftr1_norm);           
            ftr_dist=zeros(batch1,obj.n_ftrs2);
            for j=1:batch2:obj.n_ftrs2
                 batch2 = min(batch2,obj.n_ftrs2-j+1);
                 disp(['Ftr1: ' num2str(st_ftr1) ':' num2str(st_ftr1+batch1-1) ' Ftr2: ' num2str(j) ':' num2str(j+batch2-1)]);
                 ftr2=double(obj.mat_ftr2.ftr(j:j+batch2-1,:));
                 ftr2_norm = sqrt(sum(ftr2.^2, 2));
                 ftr2= bsxfun (@rdivide, ftr2, ftr2_norm);
                 dist = pdist2(ftr1,ftr2);
                 ftr_dist(:,j:j+batch2-1) = dist;  
            end            
        end
           
        function [ftr_dist] = calc_dist_selIdx(obj, st_ftr1, ftr1_idx, ftr2_idx, batch1, batch2)
            ftr1=double(obj.mat_ftr1.ftr(st_ftr1:st_ftr1+batch1-1,:));
            ftr1=ftr1(ftr1_idx(st_ftr1:st_ftr1+batch1-1),:);
            ftr1_norm=sqrt(sum(ftr1.^2, 2));
            ftr1=bsxfun (@rdivide, ftr1, ftr1_norm);        
            batch1_sel=sum(ftr1_idx(st_ftr1:st_ftr1+batch1-1)); 
            n_ftrs2sel=sum(ftr2_idx); 
            ftr_dist=zeros(batch1_sel, n_ftrs2sel);
            k=1;
            for j=1:batch2:obj.n_ftrs2
                 batch2 = min(batch2,obj.n_ftrs2-j+1);
                 disp(['Ftr1: ' num2str(st_ftr1) ':' num2str(st_ftr1+batch1-1) ' Ftr2: ' num2str(j) ':' num2str(j+batch2-1)]);
                 ftr2=double(obj.mat_ftr2.ftr(j:j+batch2-1,:));
                 ftr2=ftr2(ftr2_idx(j:j+batch2-1),:);
                 ftr2_norm=sqrt(sum(ftr2.^2, 2));
                 ftr2=bsxfun (@rdivide, ftr2, ftr2_norm);
                 dist=pdist2(ftr1,ftr2);
                 batch2_sel=sum(ftr2_idx(j:j+batch2-1)); 
                 disp(['Kcca kernel:' num2str(k) '-' num2str(k+batch2_sel-1)]);
                 if batch2_sel>0
                     ftr_dist(:,k:k+batch2_sel-1) = dist;
                     k=k+batch2_sel;
                 end    
            end            
        end
        
    end
    
    methods (Static)
        
        function [dist]=calc_pdist2(ftr1,ftr2)
            ftr1_norm=sqrt(sum(ftr1.^2, 2));
            ftr1=bsxfun (@rdivide, ftr1, ftr1_norm);           
            ftr2_norm = sqrt(sum(ftr2.^2, 2));
            ftr2=bsxfun (@rdivide, ftr2, ftr2_norm);
            dist = pdist2(ftr1,ftr2);
        end
        
    end
end

