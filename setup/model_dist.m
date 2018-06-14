function model_dist( fFtr1,fFtr2,fModel,fNN,batch1,batch2,isTest )

fFtr1 = matfile(fFtr1);
fFtr2 = matfile(fFtr2);
n_ftrs1 = fFtr1.n_ftrs;
ftrDist = FtrDist(fFtr1, fFtr2);
fModel = matfile(fModel,'Writable',true);
%{
maxDist=-1;
sumDist=0;

% Compute distance
for i=1:batch1:n_ftrs1
    batch1 = min(batch1,n_ftrs1-i+1);
    tic;
    dist=ftrDist.calc_dist(i, batch1, batch2);    
    maxDist = max(maxDist,max(max(dist)));
    sumDist=sumDist+sum(sum(dist));
    toc;
end;
fModel.maxDist=maxDist;
fModel.sumDist=sumDist;
%}

% Compute NN
%%{
maxDist = fModel.maxDist;
%matNN = matfile(fNN,'Writable',true);
K=5;
%matNN.nn=zeros(K,n_ftrs1,'uint32');
test_sortBySim = zeros(n_ftrs1,2);
test_sortBySim(:,1)= (1:n_ftrs1)';
for st_idx=1:batch1:n_ftrs1
    batch1 = min(batch1,n_ftrs1-st_idx+1);
    ed_idx = st_idx+batch1-1;
    tic;
    dist=ftrDist.calc_dist(st_idx, batch1, batch2);    
    dist=dist/maxDist;
    disp(['NN:' num2str(st_idx) '-' num2str(ed_idx)]);
    if isTest
       [~,ND]=getNN(dist,batch1,K,0);
    else
       [~,ND]=getNN(dist,batch1,K,st_idx);
    end   
    test_sortBySim(st_idx:ed_idx,2)=mean(ND)';
    %matNN.nn(:,st_idx:ed_idx)=NN;
    toc;
end;
test_sortBySim = sortrows(test_sortBySim,2);
fModel.test_byL2Sim=test_sortBySim(:,1);
%%}

end

