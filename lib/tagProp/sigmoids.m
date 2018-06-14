function ab = sigmoids( Ypred, AN, NW, AW, params )

if ~isfield(params,'verb'),     params.verb=0; end
if ~isfield(params,'ss'),       params.ss=0; end
if ~isfield(params,'iters'),    params.iters=100; end

[I W]   = size(Ypred);
ab      = zeros(2,W);
Labels  = zeros(I, W);

ii = 0;
for i=1:I
    if NW(i)>0
        ii = ii + (1:NW(i));
        Labels(i,AN(ii)+1) = 1;
        ii = max(ii);
    end
end

if params.verb, fprintf('--> word %3d',0); end
for w=1:W
    if params.verb, fprintf('\b\b\b%3d',w); end
    data    = [Ypred(:,w) ones(I,1)];
    labels  = 2*Labels(:,w)-1;
    weights = AW(w,:)';
       
    ab(:,w) = minimize(ab(:,w), 'logsigmoid',params.iters,params.verb,data,labels,weights);
end
if params.verb, fprintf('\n'); end


