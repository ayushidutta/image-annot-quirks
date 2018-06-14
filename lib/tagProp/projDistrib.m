function [x fv e]  = ProjDistrib(y)
%
%

y   = y(:);

N   = length(y);

H   = 2*eye(N);
f   = -2*y;

Aeq   = ones(1,N); % equality constraint
beq   = 1;         % equality constraint

A = [];
b = [];

lb  = zeros(N,1);
ub  = [];

options = optimset('LargeScale','off','Display','off');

[x fv e] = quadprog(H,f,A,b,Aeq,beq,lb,ub,[],options);

fv = fv + y'*y;

x = max(0,x);x = x /sum(x);