function [x h] = ProjGradArmijo(x0,fg,pr,params,varargin)
%
% minimize f(x) using projected gradient method
% using generalized Armijo stepsize selection
%
% x0 : starting point for optimization
% fg : function that computes f(x) and f'(x)
% pr : function that projects x to the convex constraint set
%
% params : struct
%  .iters : maximum number of iterations
%  .beta, .sigma, .s : parameters of the generalized Armijo step size selection
% 0<beta<1, 0<sigma<1, 0<s
%
%

iters = params.iters;
verb  = params.verb;
beta  = params.beta;
sigma = params.sigma;
tol   = params.tol;
ftol  = params.ftol; % convergence cirterion on the function value change after performing the gradient step

x   = x0;
a   = 1;

Nfe = 0; % nr of function evaluations

h = zeros(length(x),iters);
for i = 1:iters

    h(:,i)    = x;

    [f,g]     = feval(fg,x,varargin{:}); Nfe = Nfe + 1;
    if  (i>1) && (abs(f-f_old)<ftol)
        if verb
            fprintf('--> Function value decreased less than %f, converged.\n',ftol);
        end
        break
    end
    f_old     = f;      

    if verb, fprintf('iter %3d, %3d evaluations of %s + %s, f=%f ',i,Nfe,fg,pr,f); end
    
    xm        = feval(pr,x-a*g);
    [fnew tmp] = feval(fg, xm,varargin{:});Nfe = Nfe + 1;fprintf('.');
    reduction = f - fnew;
    bound     = sigma *  sum( (x(:)-xm(:)).^2 ) / a;
    condition = reduction >= bound;
    pwr       = 1;

    if condition;   % increase the step size if condition already holds
        pwr = -1;
        condition = false;
    end

    while ~condition
        a         = a * beta^pwr;
        xmOLD     = xm;
        xm        = feval(pr,x-a*g);
        [fnew tmp] =    feval(fg, xm,varargin{:});Nfe = Nfe + 1;
        reduction = f - fnew;
        if verb,fprintf('.'); end
        bound     = sigma *  sum( (x(:)-xm(:)).^2) / a;
        condition = reduction >= bound;
        if pwr==-1;
            condition = ~condition;      
            if all(xm==xmOLD); condition=1;end
        end
        if (i>1) && (max(abs(xm-xmOLD))<=tol); condition=1;end
    end

    if pwr==-1;
        a = a*beta;
        xm = feval(pr,x-a*g);
    end

    if max(abs(x-xm))==0            % changed to allow for the break also when verb==0, JJV & TM 7/3/09
        if verb; disp('*'), end
        break;
    end
    x = xm;

    if verb,fprintf('\n');end
    if max(abs(x-h(:,i)))<tol 
        if verb
        fprintf('--> Converged with tolerance %f\n',tol);
        end
        break
    end
end

[f tmp] = feval(fg,x,varargin{:}); Nfe = Nfe + 1; if verb,fprintf('.'); end
if verb, fprintf('finished %3d function evaluations, f=%f\n',Nfe,f); end
    
h = h(:,1:i);
h(:,i+1) = x;
