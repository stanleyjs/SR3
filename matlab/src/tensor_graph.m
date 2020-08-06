function [Phi] = tensor_graph(X,kNN,params)

% Phi is edge incidence matrix 
% kNN is an array where kNN(i) is the desired neighborhood size for mode i
% of X
    [p,~,~] = fileparts(mfilename('fullpath'));
    defaults.paths.tensor_toolbox = [p '/tensor_toolbox'];
    defaults.paths.rann = [p '/RANN'];
    defaults.low_rank = 6;
    defaults.approx = 1;
    defaults.adaptive  = 0;
    defaults.thresh = 0.1;
    defaults.connect = 1;
    if ~exist('params')
        params = struct();
    end
    params = default_param_struct(params,defaults);
    if params.approx
        check_rann(params.paths.rann);
    end
    
    %%%%%%%% bug with spten - disabling for now
    %X = sparse_or_dense_tensor(X, false, SR3.params.paths.tensor_toolbox);
    %%%%%%%%%%
    X = tensor(X);
    sp = false;
    
    if sp
        matfun = @(x,c) sptenmat(x,c);
    else
        matfun = @(x,c) tenmat(x,c);
    end
    modes = ndims(X);
    Phi = cell(modes,1);
    paramsbak = params;
    for ell = 1:modes
        mode_ell_matrix = matfun(X, ell);
        params.k = kNN(ell);
        if params.k >= size(mode_ell_matrix,1)
            sprintf('Invalid k supplied to tensor_graph')
            params.k = floor(size(mode_ell_matrix,1)/3);
        end
        if numel(paramsbak.adaptive)>1
            params.adaptive = paramsbak.adaptive(ell); 
            params.thresh = paramsbak.thresh(ell);
        end
        [phi] = calculate_knn_graph(double(mode_ell_matrix), params);
        Phi{ell} = phi;
    end
return;
