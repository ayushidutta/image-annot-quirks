function [f, g] = logsigmoid(v,data,labels,Weights)
%
% [f, g] = logsigmoid(v,data,labels,Weights)
% calculate the negative log-likelihood of sigmoid (and its gradient) of the labels (-1, +1) 
% a weighted log-likelihood is used: the log-likelihood of each outcome is
% weighted by the corresponding weight
%
% v      (D x 1) : parameter vector 
% data   (N x D) : each row represents a D dimensional data point
% labels (N x 1) : labels: (-1, +1)
% Weights(N x 1) : optional weighting of data points
% 
LogEps = 1e-15;

N   = size(data,1);

if nargin<4; 
    Weights = ones(N,1);
end

p       = sigmoid( labels .* (data * v) );

f       = Weights' * log( p + LogEps );
g       = data' *(labels.*Weights.*(1-p));

f = -f; % we minimize the -negative- log-likelihood
g = -g;

if ~all(isreal(g)); keyboard;end
