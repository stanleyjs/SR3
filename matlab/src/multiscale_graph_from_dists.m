function [Phi] = multiscale_graph_from_dists(Xdists,kNN,params)

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

    modes = numel(Xdists);
    Phi = cell(modes,1);
    paramsbak = params;
    for ell = 1:modes
        dists_ell = Xdists{ell};
        params.k = kNN(ell);
        if params.k >= size(dists_ell,1)
            sprintf('Invalid k supplied to multiscale_graph')
            params.k = floor(size(dists_ell,1)/3);
        end
        if numel(paramsbak.adaptive)>1
            params.thresh = paramsbak.thresh(ell);
        end
        if params.adaptive
            Phi{ell} = kernel_edges_from_dists(dists_ell,params);
        else
            [~,ix] = sort(dists_ell);
            nbrs = ix(:,2:params.k+1);
            Phi{ell} = graph_from_knn(nbrs, params);
        end
    end
return;
