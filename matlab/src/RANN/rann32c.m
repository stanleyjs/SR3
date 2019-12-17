% RANN32C -  Random Approximate nearest neighbors -  Jones, Osipov, Rokhlin
% Wraps rann32c.mexa64.
% X is an m x n data matrix.  Finds the nearest neighbors on the columns,
% i.e. size(Idx) = [k,n]
% PARAMETERS and DEFAULTS:
% params.k = 5, number of nearest neighbors
% params.numit = 5, number of iterations
% params.isuper = true, supercharging
% params.istat = false, debugging printouts. 
% Jay S. Stanley III June 2019
function [Idx,Dis] = rann32c(X,params)
    defaults.numit = 5;
    defaults.isuper = 1;
    defaults.istat = 0;
    defaults.k = 5;
    
    if nargin == 1 %defaults
        params = defaults;
    else
        params = default_param_struct(params, defaults);
    end
    
    [~,n] = size(X);
        
    assert(params.k<n,... %%%If this assertion is not here matlab will crash
        'Supplied k = %i >= %i = n columns. RANN32C requires k < n', params.k,n) 

    [Idx, Dis] = rann32cm(X,params.k,params.numit,params.isuper,params.istat);
end