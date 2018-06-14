function [ model loglik ] = tagprop_learn( NN, ND, Y, varargin )
% [ params loglik ] = tagprop_learn( NN, ND, Y, options )
% Learn TagProp and output its parameters
%
% NN : a (K x N) matrix with nearest neighbor indices between 1 and N.
% ND : a (D x K x N) vector with D distance values to the K nearest neighbors
%      of each of the N data points. Can be [] if 'type' is 'rank'.
% Y  : a (V x N) logical annotation matrix. Can be sparse. In case of
%      the 'multiclass' version, this should be the 1-of-K coding of
%      the labels.
%
% Options :
%  'class' : 'multiclass' or 'multilabel' (default: 'multilabel').
%            'multiclass' not implemented yet!
%  'type'  : 'rank' or 'dist' (default: 'rank').
%            Rank-based or distance-based weights.
%  'weighted' : logical scalar (default: true).
%               Balance positive and negative annotations.
%  'sigmoids' : logical scalar (default: false).
%               Modulate the output score with sigmoids.
%  'init' : 'flat', 'linear' or a model struct (default: 'flat').
%           'flat' (only with sigmoids) starts with sigmoid estimation
%           'linear' start with weight parameter estimation
%           model: start from previous optimization
%  'iterO' : number of outer iter (default 5)
%  'iterI' : number of inner iter (default 100)
%  'verb' : verbosity (from 0 to 3) (default 0)
%
% Output :
% model : this struct can be used in tagprop_learn (as the 'options'
%         argument) or tagprop_predict (as the 'model' argument)
% loglik : log-likelihood value sequence during optimization
%
% Matthieu Guillaumin, 18/10/2010


p=inputParser;
%%% Required arguments
% The nearest neighbor matrix is 2D and numeric within 1..N
p.addRequired('NN', @(X) ndims(X)==2 && isnumeric(X) && min(X(:))>0 && max(X(:))<=size(X,2) && size(X,2)==size(Y,2) && size(X,1)>0);
% The neighbor distance matrix is either empty ('rank' case), or 3D and numeric
p.addRequired('ND', @(X) isempty(X) || (ndims(X)==3 && isnumeric(X) && size(X,2) == size(NN,1) && size(Y,2) == size(X,3) && size(X,1)>0));
% The annotation matrix has as many columns as NN or ND
p.addRequired('Y', @(X) ndims(X)==2 && (size(X,2) == size(NN,2) || size(X,2) == size(ND,3)));
%%% Optional arguments
p.addOptional('options', struct(), @isstruct);
%%% Parameters
p.addParamValue('class', 'multilabel', @(X) strcmpi(X,'multilabel'));
p.addParamValue('type', 'rank', @(X) strcmpi(X,'rank') || strcmpi(X,'dist') );
p.addParamValue('weighted', true, @islogical);
p.addParamValue('sigmoids',false, @islogical);
p.addParamValue('init', 'flat',  @(X) strcmpi(X,'flat') || strcmpi(X,'linear') );
p.addParamValue('iterO',5, @isnumeric);
p.addParamValue('iterI',100, @isnumeric);
p.addParamValue('verb',0, @isnumeric);
%%% Validate input
p.parse(NN,ND,Y,varargin{:});
%%% Check if options are provided as a struct or not.
if isempty(fieldnames(p.Results.options)),
  options = rmfield(p.Results,{'options','NN','ND'});
else
  options = p.Results.options;
end
%%% Set NN and ND for the chosen type
NN=NN-1; % zero-based indexing for neighbors
if strcmp(options.type,'rank')
    ND=[]; % important to make this empty for the c code.
elseif isempty(ND)
    error('Neighbor distance matrix should not be empty!');
end
%%% Initialize with previous model or use default parameters
if isstruct(options.init),
    if options.verb, fprintf('Initializing with given parameters\n'); end
    params = options.init;
    % TODO: check if parameters are for the correct data
else
    if options.verb, fprintf('Initializing with default parameters\n'); end
    % Set data weights
    params.NW      = sum(Y,1); % number of labels for each data point
    params.nw      = sum(params.NW);
    params.I       = numel(params.NW); % number of data points
    params.W       = size(Y,1);
    if options.weighted,
        AW = Y * (1/params.nw) + (1-Y) / (params.I*params.W-params.nw);
        AW = AW / sum(AW(:)); % normalize weights to unit sum
    else
        AW = ones(size(Y));
    end
    params.AW=AW;
    switch (options.type)
        case 'dist',
            params.D            = size(ND,1); % number of different distances to combine
            params.K            = size(ND,2); % number of nearest neighbors
            params.function     = 'tagpropCmt';
            params.projection   = 'projNonNegative';
            params.lambda       = zeros(params.D,1); %1./(mean(reshape(ND,params.D,[]),2)+eps); % TODO: replace with heuristics
        case 'rank', % in fact, this is 'rexp'
            params.K            = size(NN,1);
            params.D            = 0;
            params.function     = 'tagpropCmt';
            params.projection   = 'projNoConstraints';
            params.lambda       = zeros(params.K,1);
        otherwise
            error('Unknown method %s',options.type);
    end    % check NN
    params.sigmoid  = struct('verb', options.verb>2, 'ss', 0, 'weighted', 1, 'iters', options.iterI );
    params.projgrad = struct('verb', options.verb>1, 'iters', options.iterI, 'beta', 0.1, 'sigma', 0.01, 'tol', 1e-4,'ftol',1e-6);
    params.AL = zeros(params.nw,1);
    count = 0;
    for i=1:params.I
        tmp  = find(Y(:,i));
        params.AL(count + (1:params.NW(i))) = tmp-1; % zero-based indices for labels
        count = count + params.NW(i);
    end
    %keyboard
    if strcmp(options.init,'linear') || ~options.sigmoids,        
        if options.verb, fprintf('Learning linear %s model\n',options.type); end                
        params.lambda = projGradDescentArmijo(params.lambda, params.function, params.projection, params.projgrad, ...
                                              params.AL, params.NW, params.AW, NN, ND);
    end
end

%%% Prepare outer iterations
iters       = options.iterO * (options.sigmoids>0); % no outer iteration if no sigmoids
loglik      = zeros(iters+1,2);

%%% If sigmoids, first set the sigmoids parameters
if options.sigmoids,
    % linear prediction from neighbors for train annotations
    Y = tagpropCmt(params.lambda,params.AL,params.NW,params.AW,NN,ND)';     
    % estimate sigmoid parameters under initial weighted predictions
    if options.verb, fprintf('\nFitting sigmoids\n'); end                
    params.ab = sigmoids(Y,params.AL,params.NW, params.AW, params.sigmoid);
    % get initial likelihood value
    [loglik(1,1) tmp] = tagpropCmt(params.lambda,params.AL,params.NW,params.AW,NN,ND,params.ab(1,:),params.ab(2,:));     
else
    % get initial likelihood value
    [loglik(1,1) tmp] = tagpropCmt(params.lambda,params.AL,params.NW,params.AW,NN,ND);
end

%%% Alternate optimization of weights and sigmoids
for outer_iter = 1:iters
    % re-estimate weight parameters under fixed sigmoids
    if options.verb, fprintf('Learning %s model\n',options.type); end                
    params.lambda = projGradDescentArmijo(params.lambda, params.function,params.projection,params.projgrad,params.AL,params.NW,params.AW,NN,ND,params.ab(1,:),params.ab(2,:));        
    Y = tagpropCmt(params.lambda,params.AL,params.NW,params.AW,NN,ND)';         
    [loglik(outer_iter,2) tmp] = tagpropCmt(params.lambda,params.AL,params.NW,params.AW,NN,ND,params.ab(1,:),params.ab(2,:));     
        
    % re-estimate sigmoid parameters under fixed weighted predictions
    if options.verb, fprintf('\nFitting sigmoids\n'); end                
    params.ab = sigmoids(Y,params.AL,params.NW, params.AW, params.sigmoid);    
    [loglik(outer_iter+1,1) tmp] = tagpropCmt(params.lambda,params.AL,params.NW,params.AW,NN,ND,params.ab(1,:),params.ab(2,:));     
end

%%% Output model/options struct
if isfield(params,'loglik'),
    % append evaluated log-likelihoods to previously evaluated ones
    params.loglik = [ params.loglik; loglik ];
else
    params.loglik = loglik;
end
model = options;
model.init = params;

