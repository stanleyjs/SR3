%%% CALCULATE_KNN_GRAPH
%   Compute a connected graph with the rows of X as the nodes.
%   Returns a sparse incidence matrix for X. 
%   PARAMS and DEFAULTS:
%     params.k = 5; % kNN
%     params.low_rank = 6; % number of singular vectors to take. if 0, then no LRA.
%     params.approx = 1; % Use RANN, requires rann32cm.mex and rann32c
%       wrapper
%     params.rann.numit = 5; 
%     params.rann.isuper = 1;
%     params.rann.istat = 0;
%     params.knnsearch.<key> accepts parameters for the builtin KNNSEARCH
%   Jay S. Stanley III June 2019 
function [Phi] = calculate_knn_graph(X,params)
    
    defaults.k = 5; % kNN
    defaults.low_rank = 6; % number of singular vectors to take. if 0, then no LRA.
    defaults.approx = 1; % Use RANN
    defaults.thresh = 0.1;
    defaults.connect = 0;
    defaults.adaptive = 0;
    defaults.rann.numit = 5;
    defaults.rann.isuper = 1;
    defaults.rann.istat = 0;
    [p,~,~] = fileparts(mfilename('fullpath'));
    defaults.paths.rann = [p '/RANN'];
    if nargin==1
        params.rann = struct();
        params.knnsearch = struct();
    end
    params = default_param_struct(params,defaults);

    [n,~] = size(X);
    %%%%Here we parse the knn search function to be used.
    if params.approx
        check_rann(params.paths.rann);
        params.rann.k = params.k;
        params.knnparams = default_param_struct(params.rann,defaults.rann);
        knnfun = @(x) rann32c(x',params.knnparams)'; %RANN32c treats each x as a column vector.
    else
        params.knnsearch.k = params.k+1;
        params.knnparams = default_param_struct(params.knnsearch,...
                                                defaults.knnsearch); 
        params.knnparams = struct2kwargs(params.knnparams); %KNNSEARCH takes kwargs
        knnfun = @(x) knnsearch(x,x,params.knnparams{:});
        slice =  @(A) A(:,2:end);  %remove the self-index from the output of knnsearch
        knnfun = @(x) slice(knnfun(x));
    end
    
    %%%Perform knn on our data (possibly LRA of the data)
    if params.low_rank
        [u,s,~] = svds(X,params.low_rank);
        X = u*s;
    end
    [idx] = knnfun(X);
    idx = double(idx);
    %%%Compute the edge set
    if ~params.adaptive
        Phi = graph_from_knn(idx,params);

    else
        dists = squareform(pdist(X));
        Phi = kernel_edges_from_dists(dists,params);
    end
    

    return;
end