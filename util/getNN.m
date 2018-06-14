function [ NN,ND ] = getNN( dist,N,K,st_tr)
%GETNN Gets nearest neighbours of each N
       %st_tr : Start train index / 0 otherwise

NN=zeros(K,N,'uint32');
ND=zeros(K,N);
for i=1:N
     if(st_tr>0)
         dist(i,i+st_tr-1)=inf;
     end    
     for k=1:K
        [val,idx] = min(dist(i,:));
        NN(k,i)=idx;
        ND(k,i)=val;
        dist(i,idx)=inf;
     end;   
end;

end

