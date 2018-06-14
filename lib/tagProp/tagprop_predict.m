function Y = tagprop_predict(NN, ND, model)

params = model.init;
AW = ones(params.W,params.I);

if strcmp(model.type,'rank'),
  ND=[];
  if ~(isnumeric(NN) && size(NN,1)==params.K && max(NN(:))>0 && max(NN(:))<=params.I)
    error('Invalid nearest neighbor matrix');
  end
else
  ND=ND(:);
  if ~(isnumeric(ND) && rem(numel(ND),params.K*params.D)==0),
    error('Invalid neighbor distance matrix')
  end
end
NN=NN-1;  % zero-based indexing

if isfield(params,'ab') && model.sigmoids, % model has sigmoids
    Y = tagpropCmt(params.lambda(:),params.AL,params.NW,AW,NN,ND,params.ab(1,:)',params.ab(2,:)')'; 
else % without sigmoids
    Y = tagpropCmt(params.lambda(:),params.AL,params.NW,AW,NN,ND)'; 
end
