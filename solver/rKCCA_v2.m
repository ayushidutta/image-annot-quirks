classdef rKCCA_v2
    %RKCCA KCCA projection
    
    properties (Constant=true)
        eta = 30;
        kapa = 0.1;
        sl = 0;
    end 
    
    properties
        fTrainAnnot
        fModel
        fTrainFtr
        fTestFtr
        fTrainKcca
        fTestKcca
    end
    
    methods 
        
         function [obj] = rKCCA_v2(dData,dModel,fTrainFtr,fTestFtr,fTrainAnnot,fModel,fTrainKcca,fTestKcca)
            obj.fTrainAnnot = fullfile(dData, fTrainAnnot);
            dModel = fullfile(dData,dModel);                        
            obj.fTrainFtr = fullfile(dModel,fTrainFtr);
            obj.fTestFtr = fullfile(dModel,fTestFtr);
            obj.fTrainKcca = fullfile(dModel,fTrainKcca);
            obj.fTestKcca = fullfile(dModel,fTestKcca);
            obj.fModel = fullfile(dModel, fModel);             
        end
        
        function train(obj,batch1,batch2)
            fFtr = matfile(obj.fTrainFtr);
            ftrDist = FtrDist(fFtr, fFtr);
            n_train = fFtr.n_ftrs;
            matModel = matfile(obj.fModel,'Writable',true);
            maxDist=matModel.maxDist;
            meanDist=matModel.sumDist/(n_train * n_train);
            tr_kernel = zeros(n_train,n_train);
            for st_idx=1:batch1:n_train
                batch1 = min(batch1,n_train-st_idx+1);
                ed_idx = st_idx+batch1-1;
                curr_dist = ftrDist.calc_dist(st_idx, batch1, batch2);    
                curr_dist = curr_dist/maxDist;
                curr_dist = exp(-curr_dist/meanDist);
                tr_kernel(st_idx:ed_idx,:)= curr_dist;
            end; 
            Y_tr = load(obj.fTrainAnnot);
            tr_kernelY = Y_tr*Y_tr';            
            disp('Learning projections. (swap)'); % Full training
            [~,nbeta,r,~,~] = kcanonca_reg_ver2(tr_kernelY,tr_kernel,rKCCA.eta,rKCCA.kapa,rKCCA.sl);
            kcca=struct;
            kcca.nbeta=nbeta;
            kcca.r=r;
            matModel.kcca=kcca;
            disp('Projecting train features');
            disp([size(tr_kernel) size(nbeta) size(r)]);
            kcca_tr = tr_kernel*nbeta*diag(r);
            clear tr_kernel;
            fKcca = matfile(obj.fTrainKcca,'Writable',true);
            fKcca.ftr=kcca_tr;
            fKcca.n_ftrs=n_train;
        end
        
        function project(obj,batch1,batch2)
            fFtr1 = matfile(obj.fTestFtr);
            fFtr2 = matfile(obj.fTrainFtr);
            ftrDist = FtrDist(fFtr1, fFtr2);
            n_test = fFtr1.n_ftrs;
            n_train = fFtr2.n_ftrs;
            matModel = matfile(obj.fModel);
            maxDist=matModel.maxDist;
            meanDist=matModel.sumDist/(n_test * n_train);
            kernel = zeros(n_test,n_train);
            for st_idx=1:batch1:n_test
                batch1 = min(batch1,n_test-st_idx+1);
                ed_idx = st_idx+batch1-1;
                curr_dist = ftrDist.calc_dist(st_idx, batch1, batch2);    
                curr_dist = curr_dist/maxDist;
                curr_dist = exp(-curr_dist/meanDist);
                kernel(st_idx:ed_idx,:)= curr_dist;
            end; 
            kcca=matModel.kcca;
            nbeta=kcca.nbeta;
            r=kcca.r;
            disp('Projecting features');
            kcca_ftrs = kernel*nbeta*diag(r);
            clear kernel;
            fKcca = matfile(obj.fTestKcca,'Writable',true);
            fKcca.ftr=kcca_ftrs;
            fKcca.n_ftrs=n_test;
        end
        
        function modelSubset(obj, tr_ratio, batch1, batch2)
            fFtr = matfile(obj.fTrainFtr);
            ftrDist = FtrDist(fFtr, fFtr);
            n_train = fFtr.n_ftrs;
            matModel = matfile(obj.fModel,'Writable',true);
            [trainIdx,~,~] = dividerand(n_train,tr_ratio,0,1-tr_ratio);
            allTrainIdx=zeros(1,n_train);   
            allTrainIdx(trainIdx)=1;
            trainIdx=logical(allTrainIdx);
            matModel.trKernelIdx=trainIdx;
            maxDistSplit=0;
            sumDistSplit=0;
            for st_idx=1:batch1:n_train
                batch1 = min(batch1,n_train-st_idx+1);
                dist = ftrDist.calc_dist_selIdx(st_idx, trainIdx, trainIdx, batch1, batch2); 
                maxDistSplit=max(maxDistSplit,max(max(dist)));
                sumDistSplit=sumDistSplit+sum(sum(dist));
            end 
            matModel.maxDistKernel=maxDistSplit;
            matModel.sumDistKernel=sumDistSplit;
        end
        
        function trainSubset(obj,batch1,batch2)
            fFtr = matfile(obj.fTrainFtr);
            ftrDist = FtrDist(fFtr, fFtr);
            n_train = fFtr.n_ftrs;
            matModel = matfile(obj.fModel,'Writable',true);
            maxDist=matModel.maxDistKernel;
            trainIdx=matModel.trKernelIdx;
            n_trSplit=sum(trainIdx);
            meanDist=matModel.sumDistKernel/(n_trSplit * n_trSplit);
            st_trIdx=1;
            tr_kernel = zeros(n_trSplit,n_trSplit);
            for st_idx=1:batch1:n_train
                batch1 = min(batch1,n_train-st_idx+1);
                curr_dist = ftrDist.calc_dist_selIdx(st_idx, trainIdx, trainIdx, batch1, batch2); 
                curr_dist=curr_dist/maxDist;
                ed_trIdx=st_trIdx+sum(trainIdx(st_idx:st_idx+batch1-1))-1;
                disp(['Kcca kernel:' num2str(st_trIdx) '-' num2str(ed_trIdx)]);
                tr_kernel(st_trIdx:ed_trIdx,:)= exp(-curr_dist/meanDist); 
                st_trIdx=ed_trIdx+1;
            end 
            clear curr_dist;
            Y_tr = load(obj.fTrainAnnot);
            Y_tr = Y_tr(trainIdx,:);
            tr_kernelY = Y_tr*Y_tr';            
            disp('Learning projections. (swap)'); % Full training
            [~,nbeta,r,~,~] = kcanonca_reg_ver2(tr_kernelY,tr_kernel,rKCCA.eta,rKCCA.kapa,rKCCA.sl);
            kcca=struct;
            kcca.nbeta=nbeta;
            kcca.r=r;
            matModel.kcca=kcca;
        end
           
        function trainSubset1(obj)
            fFtr = matfile(obj.fTrainFtr);
            matModel = matfile(obj.fModel,'Writable',true);
            trainIdx=matModel.trKernelIdx;
            n_trSplit=sum(trainIdx);
            ftr=fFtr.ftr(trainIdx,:);
            ftr_norm=sqrt(sum(ftr.^2, 2));
            ftr=bsxfun (@rdivide, ftr, ftr_norm);
            dist=pdist2(ftr,ftr);
            maxDist=max(max(dist));
            sumDist=sum(sum(dist));
            meanDist=sumDist/(n_trSplit * n_trSplit);
            disp(['Max' 'Sum' num2str(maxDist) num2str(sumDistl)]);
            matModel.maxDistKernel=maxDist;
            matModel.sumDistKernel=sumDist;
            dist=dist/maxDist;
            dist=exp(-dist/meanDist);
            clear dist;
            Y_tr = load(obj.fTrainAnnot);
            Y_tr = Y_tr(trainIdx,:);
            tr_kernelY = Y_tr*Y_tr';            
            disp('Learning projections. (swap)'); % Full training
            [~,nbeta,r,~,~] = kcanonca_reg_ver2(tr_kernelY,dist,rKCCA.eta,rKCCA.kapa,rKCCA.sl);
            kcca=struct;
            kcca.nbeta=nbeta;
            kcca.r=r;
            matModel.kcca=kcca;
        end
        
        function projectFtrSubset(obj,fFtr1,fKcca,batch1,batch2)  
            fFtr2 = matfile(obj.fTrainFtr);
            ftrDist = FtrDist(fFtr1, fFtr2);
            n_test = fFtr1.n_ftrs;
            matModel = matfile(obj.fModel);
            maxDist=matModel.maxDistKernel;
            trainIdx=matModel.trKernelIdx;
            n_trSplit=sum(trainIdx);
            meanDist=matModel.sumDistKernel/(n_trSplit * n_trSplit);
            kernel = zeros(n_test,n_trSplit);
            for st_idx=1:batch1:n_test
                batch1 = min(batch1,n_test-st_idx+1);
                ed_idx = st_idx+batch1-1;
                curr_dist = ftrDist.calc_dist_selIdx(st_idx, logical(ones(1,n_test)), trainIdx, batch1, batch2); 
                curr_dist = curr_dist/maxDist;
                curr_dist = exp(-curr_dist/meanDist);
                kernel(st_idx:ed_idx,:)= curr_dist;
            end; 
            kcca=matModel.kcca;
            nbeta=kcca.nbeta;
            r=kcca.r;
            disp('Projecting features');
            kcca_ftrs = kernel*nbeta*diag(r);
            clear kernel;
            fKcca = matfile(fKcca,'Writable',true);
            fKcca.ftr=kcca_ftrs;
            fKcca.n_ftrs=n_test;
        end
        
        function projectSubset(obj,batch1,batch2)
            % Project train features
            fFtr1 = matfile(obj.fTrainFtr);            
            obj.projectFtrSubset(fFtr1,obj.fTrainKcca,batch1,batch2);
            % Project test features
            fFtr1 = matfile(obj.fTestFtr);
            obj.projectFtrSubset(fFtr1,obj.fTestKcca,batch1,batch2);            
        end
        
    end
    
end

