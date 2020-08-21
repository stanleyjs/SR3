function [output, gammas, ratios, magnitudes] = SR3_simplex2(X, varargin)
    [X,phi,convexparams,SR3,verbose] = init_and_parse(X,varargin{:});
    Nratios = convexparams.Nratios;
    [gammas,ratios,magnitudes] = convexratios2(convexparams.dims,...
        convexparams.max_gamma,Nratios,convexparams.Nmagnitudes,convexparams.min_mag,convexparams.max_mag);
    gt_ratios = unique(ratios,'rows');
    output = struct();
    
    gt_ratios = gt_ratios(any(gt_ratios>0,2),:);
    
    % init output strcut to correct size
    n_iters = size(gt_ratios,1);
    output(n_iters).ratio = gt_ratios(n_iters,:);
    mask = ratios == gt_ratios(n_iters,:);
    mask = all(mask,2);
    gamj = gammas(mask,:);
    output(n_iters).gammas = gamj;
    output(n_iters).magnitudes = magnitudes(mask);
    output(n_iters).U = cell(1,size(gamj,1));
    output(n_iters).V = cell(1,size(gamj,1));
    output(n_iters).F = cell(1,size(gamj,1));
    
    parfor j = 1:size(gt_ratios,1)%:-1:1
        output(j).ratio = gt_ratios(j,:);
        mask = ratios == gt_ratios(j,:);
        mask = all(mask,2);
        gamj = gammas(mask,:);
        output(j).gammas = gamj;
        output(j).magnitudes = magnitudes(mask);
        output(j).U = cell(1,size(gamj,1));
        output(j).V = cell(1,size(gamj,1));
        output(j).F = cell(1,size(gamj,1));
        fprintf('%d conifg, starting inner loop\n',j)
        for kk = 1:1:size(gamj,1)
            gamma = gamj(kk,:);
            [results,~,jj] = SR3_tensor(X,phi,SR3,gamma);
            output(j).U{kk} = results.output.U;
            output(j).V{kk} = results.output.V;
            output(j).F{kk} = results.output.F;
            output(j).iter{kk} = jj;
            fprintf('%d conifg, finished inner loop %d\n',j,kk)
        end
        output(j).SR3 = results;
    end
end

function [X, phi, convexparams, SR3,verbose] = init_and_parse(X,varargin)
    %collate and parse input parameters
    p = inputParser;
    %custom argument checkers

    %valid prox functions
    prox_flake = {@snowflake, @flakeprox};
    prox_log = {@log, @logprox};
    prox_l2 = {@vecnorm, @l2_prox};

    %defaults for optionals
    default.phi  = false;
    default.SR3.prox = prox_flake;
    default.SR3.nu = 1;
    %default.SR3.gamma = 1;
    default.SR3.missing_data = [];
    default.SR3.params.tolF = 1e-6;
    default.SR3.params.maxiter = 1000;
    default.SR3.params.epsilon = 1e-8;
    default.SR3.params.verbose = false;
    default.SR3.params.warnings = false;
    default.SR3.params.store_updates = false;
    default.SR3.params.pcg_stop = true;
    default.SR3.graph.knn = 10;
    default.SR3.graph.params.approx = true;
    default.SR3.graph.params.low_rank = 6;
    default.SR3.solver.f = 'pcg_preconditioned';
    default.SR3.solver.params.tol = 1e-8;
    default.SR3.solver.params.maxit = 100;
    default.SR3.solver.params.verbose = false;
    [pth,~,~] = fileparts(mfilename('fullpath'));
    default.SR3.params.paths.tensor_toolbox = [pth '/tensor_toolbox'];
    default.SR3.params.paths.rann = [pth '/RANN'];
    default.convexparams.Nratios = 10;
    default.convexparams.Nmagnitudes = 10;
    default.convexparams.min_mag = 1e-6;
    default.convexparams.max_mag = 1e6;
    addRequired(p, 'X', @(x) isnumeric(x) || isa(x, 'tensor') ||isa(x,'sptensor'));
    addOptional(p, 'phi', default.phi, @(x) isnumeric(x) || islogical(x) ||iscell(x));
    addOptional(p, 'convexparams',default.convexparams, @(x) isstruct(x) || islogical(x));
    addOptional(p, 'SR3',default.SR3, @(x) isstruct(x));
    
    parse(p, X, varargin{:});
    
    out = p.Results;
    X = out.X;
    phi = out.phi;
    if ~isfield(out,'convexparams') || ~isstruct(out.convexparams)
        convexparams = default.convexparams;
    else
        convexparams = default_param_struct(out.convexparams, default.convexparams);
    end
    convexparams.dims = ndims(X);
    SR3 = out.SR3;
    verbose = out.SR3.params.verbose;
end