%Fusion Clustering Using Folded-Concave Penalties
%   [SR3, phi] = SR3_tensor(X, false, SR3) Fusion clustering of the tensor X. 
%   X : N1 x N2 x ... x Nd-1 x Nd data tensor to cluster.
%
%   Each entry phi{i} is an incidence matrix associated with each mode i of X. 
%       Let mode_L(X) be the reshape of X such that the
%       L-th mode lies on the rows and every other mode is on the columns.
%   Then phi{i} = tensor_graph(mode_i(X),SR3params.params.kNN).
%
%   Optionally, one may precompute and supply  as a cell array of this form via
%       [SR3,~] = SR3(X,phi,SR3params)
%
%   SR3 is an (optional) struct with sub-structs SR3.params and SR3.solver.
%       See below.
%
%  
%   This function iterates over a sequence of gammas, which control 
%       the strength of clustering.  At each step, it
%       makes a decision for each mode whether to increase or remain at the
%       same gamma.  It proceeds until convergence.  Different gammas are often
%       required for different modes.
%
%   The function returns the input SR3 structure with the added substruct
%       SR3.output. Its fields are
%   
%   SR3.output.U : Final converged U.  if SR3.params.store_updates ==
%   true,
%        a {1 x updates} cell array where each U{i} is the
%       clustered version of x chosen based on convergence criteria at step i
%
%   SR3.output.V : Final converged V.  if SR3.params.store_updates ==
%   true, {1 x updates} cell array, the approximate distances V{i} at each step of
%       convergence
%
%   Parameters and their defaults are
%
%   SR3.knn = 10; % the nearest neighbors to build the graph over
%
%   SR3.prox = 'snowflake'; % the prox function to use.  Accepts
%       'snowflake', 'l2', 'log', or a cell array of function handles {@fun,
%       @proxfun}
%
%   SR3.nu = 1; %approximation penalty
%
%   SR3.gamma = 1; %gamma to apply
%   SR3.params.tolF = 1e-6; % threshold for iterating a mode to the next
%                             gamma based on relative change in V
%
%   SR3.params.maxiter = 1000; maximum iterations
%   SR3.params.epsilon = 1e-8; % precision
%   SR3.params.pcg_stop = false; % halt iterations when PCG only runs for 1
%   iter.
%    SR3.params.verbose = true; % print details of iterations to standard out.
%    
%    SR3.params.store_updates = false; store each iteration
%    SR3.solver.f = 'pcg_preconditioned'; % pcg with ichol preconditioning.  
%                                           Also accepts 'factors' for stored 
%                                           full cholesky factorization, or
%                                           'pcg' for no preconditioning
%           
%    The following are parameters for the pcg solver.
%         SR3.solver.params.tol =SR3.params.store_updates 1e-8;
%         SR3.solver.params.maxit = 100;
%         SR3.solver.params.verbose = false;
% 
%    SR3.missing_data -  TRUE for where values exists, false for missing
%    entries
%
function [SR3,phi,jj] = SR3_tensor(X, varargin)
%   
    [X, phi, SR3, verbose] = init_and_parse(X,varargin{:});
    

    % Laplacian from input incidence
    if verbose
        fprintf('Computing Laplacian \n')
    end
    if contains('legendre',SR3.solver.f)
        compute_lbounds = true;
    else
        compute_lbounds = false;
        lmax = 0;
        lmin = 0;
    end
    [L,A,lmin,lmax] = tensor_incidence(phi, compute_lbounds);
    modes = ndims(X);
    
    if isa(X, 'sptensor')
        x = sptenmat(X, 1:modes); %vectorize X
    else 
        x = tenmat(X,1:modes);
    end
    N = size(x,1);
    sz = size(X);
    if ~isempty(SR3.missing_data)
        I = spdiags(SR3.missing_data, 0, N, N);
    else
        I = speye(size(L,1));
    end

    
    solver = design_solver(L, I, SR3.nu, SR3.solver, verbose,lmin,lmax,phi);

    sumV = 0;
    V0 = {};
    for k = 1:modes

        V0{k} = A{k}*x;
        sumV = sumV + double(A{k}'*V0{k});
    end
    U0 = double(x);%updateU(sumV, x);
    if SR3.params.store_updates
        for k = 1:modes
            edges =size(phi{k},1);
            dimension = max(size(V0{k}))/edges;
            SR3.output.V{1}{k} = reshape(tensor(V0{k}), [edges,dimension]);
        end
        SR3.output.U{1} = remap_tensor(U0,x);
    end
    [F0,f1,f2,f3,F_prev_mode] = objective_prox(x,U0,V0,A,SR3.nu,SR3.gamma,SR3.prox,SR3.params.epsilon);
    SR3.output.F = [F0 f1 f2 f3 cell2mat(F_prev_mode)];

    jj = 2;
    if verbose
        fprintf('\n Starting loop \n')
    end
    go = true;
    while go && jj < SR3.params.maxiter
        U_prev = U0;
        Fprev = F0;
        [V0,sumV] = updateV(U0,SR3.gamma);
        [U0,iter] = updateU(sumV,U_prev,solver);
        [F0,f1,f2,f3,mode_loss] = objective_prox(x,U0,V0,A,SR3.nu,SR3.gamma,SR3.prox,SR3.params.epsilon);
        SR3.output.F = [SR3.output.F;F0 f1 f2 f3 cell2mat(mode_loss)];
        if SR3.params.store_updates
            for k = 1:modes
                edges =size(phi{k},1);
                dimension = max(size(V0{k}))/edges;
                SR3.output.V{jj}{k} = reshape(tensor(V0{k}), [edges,dimension]);
            end
            SR3.output.U{jj} = remap_tensor(U0,x);
        end
        V_nnz = 0;
        for k = 1:modes
            V_nnz = V_nnz+nnz(double(V0{k}));
        end
        if (iter == 1 && SR3.params.pcg_stop) || abs(F0-Fprev) < SR3.params.tolF || V_nnz == 0
            go = false;
        else
            jj = jj+1;
        end
    end
    if ~SR3.params.store_updates
        SR3.output.U = remap_tensor(U0,x);
        for k = 1:modes
            edges =size(phi{k},1);
            dimension = max(size(V0{k}))/edges;
            SR3.output.V{k} = reshape(tensor(V0{k}), [edges,dimension]);
        end
    end
    if verbose
        fprintf([' ***** \n SR3 halted after %i total iterations\n'],jj) 
    end

    function [U,iter] = updateU(sumV,Uprev,solve)
        Uprev = double(Uprev);
        U = SR3.nu*I*double(x) + sumV;
        U = U-mean(U);
        if verbose
            U = solve(U,Uprev);
            iter = 1;
        else
            try
                [U,~,~,iter] = solve(U,Uprev);
            catch
                U = solve(U,Uprev);
            end
        end
    end

    function [V,sumV,outdists] = updateV(U,gam)
        v = cell(modes,1);
        sumV = 0;
        outdists = [];
        for c = 1:modes
            uij = A{c}*U;
            v{c} = uij;
            stride = size(phi{c},1);
            otherdim = max(size(uij))/stride;

            uij = reshape(tensor(uij), [stride,otherdim]);
            temp = vecnorm(double(uij)');
            temp = SR3.proxfun(temp,gam(c),SR3.params.epsilon)./temp;

            v{c}(1:end) = v{c}(1:end).*repmat(temp,1,otherdim)';
            sumV = sumV+ A{c}'*v{c};
        end
        V = v;
    end
end


function [solve] = design_solver(L, I, nu, solver, verbose,lmin,lmax,phi)
    params = solver.params;
    solver = solver.f;
 % the matrix to invert
    mat = nu.*I+L;
    if contains(solver,'chebyshev')
        A = spfun(@abs,spdiags(zeros(size(L,1),1),0,L)); % remove diagonal and take abs to
                                            % convert L to an adjacency matrix A
        g = gsp_graph(A);
        filter = @(x) 1./(nu.*1+x); 
        if contains(solver,'alt')
            g.L = mat;
            filter = @(x) 1./(x); 
        end
        g = gsp_estimate_lmax(g); %estimate interval for chebyshev
        param.order = 100;
        solve = @(b,b0) gsp_filter_analysis(g,filter,b,param); 
    elseif contains(solver,'pcg')
        solve = @(b,b0,preconditioner) ...
                pcg(mat,b,params.tol,params.maxit,preconditioner,preconditioner',b0);

        pre = contains(solver,'preconditioned');
        if verbose
            if pre, txt = 'Computing preconditioner.'; 
            else txt = 'No preconditioning.'; end
            fprintf(txt)
        end
        if pre, precond = ichol(mat); else precond = speye(size(mat,1)); end
        solve = @(b,b0) solve(b,b0,precond);
    elseif contains(solver,'legendre')
        order = 50; %add to parameters ultimately.
        [psi,lambda] = prodgraph_eigs(phi,200);
        [lambda,ix] = sort(lambda);
        psi = psi(:,ix);
        lower = min(0.1,max(lambda));%10^floor(log10(lmin)-1);
        upper = lmax;%10^ceil(log10(lmax)+1);
        [coeffs,error] = fit_legendre(@(x) 1./(nu+x), order, lower, upper);
        mat = ((L./lmax)*2)-speye(size(L,1));
        
        solve = @(b,b0) partial_legendre(mat, b, order, coeffs, psi, lambda, nu);
    else
        if verbose, fprintf('Computing cholesky factors \n'); end
        m = decomposition(mat);
        solve = @(b,b0) b\mat;
    end
end

function x = partial_legendre(mat, b, order, coeffs, psi, lambda,nu)

    bproj = psi'*b;
    borth = b-(psi*bproj);
    Htapsnu = 1./(nu+lambda);
    Htaps_bproj = bproj.*Htapsnu;

    borth_approx = matvecleg(mat,borth,order,coeffs);
    x = (psi*Htaps_bproj)+borth_approx;
end
function [X, phi, SR3,verbose] = init_and_parse(X,varargin)
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
    default.SR3.nu = 1e-3;
    default.SR3.gamma = 1;
    default.SR3.missing_data = [];
    default.SR3.params.tolF = 1e-6;
    default.SR3.params.maxiter = 1000;
    default.SR3.params.epsilon = 1e-8;
    default.SR3.params.verbose = true;
    default.SR3.params.warnings = false;
    default.SR3.params.store_updates = false;
    default.SR3.params.pcg_stop = false;
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
    default.solver.f = 'pcg_preconditioned';
    addRequired(p, 'X', @(x) isnumeric(x) || isa(x, 'tensor') ||isa(x,'sptensor'));
    addOptional(p, 'phi', default.phi, @(x) isnumeric(x) || islogical(x) ||iscell(x));
    addOptional(p, 'SR3',default.SR3, @(x) isstruct(x));
    
    parse(p, X, varargin{:});
    
    out = p.Results;

    %----parse the default struct ---- %
    SR3 = out.SR3;
    if ~isstruct(SR3)
        error('SR3 input must be a struct')
    end
    SR3 = default_param_struct(SR3, default.SR3);
    SR3.params = default_param_struct(SR3.params, default.SR3.params);
    SR3.graph = default_param_struct(SR3.graph, default.SR3.graph);
    SR3.graph.params = default_param_struct(SR3.graph.params, default.SR3.graph.params);
    SR3.params.paths = default_param_struct(SR3.params.paths, default.SR3.params.paths);
    verbose = SR3.params.verbose;
    if verbose
        fprintf('Initializing SR3 fusion clustering \n')
    end
    if SR3.params.warnings
        warning('on', 'all')
    else
        warning('off','all')
    end
    X = sparse_or_dense_tensor(X, false, SR3.params.paths.tensor_toolbox);
    %-----Graph building------%
    if ~iscell(out.phi) % compute a knn graph over X.
        dims = size(X);
        knn = SR3.graph.knn;
        if verbose
            fprintf('Computing incidence matrix \n')
            if ~isnumeric(knn)
                error('Numeric knn must be supplied to calculate distance matrix');
            end
            if length(knn) ~= length(dims)
                fprintf('Attempting to identify correct knn to match all dimensions \n')
            end
        end
        if length(knn) > length(dims)
            knn = knn(1:length(dims));
        end

        while length(knn) ~= length(dims)
            knn = [knn min([knn])];
            gate = knn >= dims(1:length(knn));
            knn(gate) = ceil(dims(gate)./5);
        end
        SR3.graph.knn = knn;
        if SR3.graph.params.approx
            check_rann(SR3.params.paths.rann);
        end
        [phi] = tensor_graph(X, SR3.graph.knn, SR3.graph.params);
    else
        phi = out.phi;            
    end
    
    %-------Prox/penalty function parsing------
    if isstring(SR3.prox) || ischar(SR3.prox)
        SR3.prox = lower(char(SR3.prox));
        switch SR3.prox
            case 'l2'
                penalty = prox_l2;
            case 'snowflake'
                penalty = prox_flake;
            case 'log' 
                penalty = prox_log;
        end
        if isstring(SR3.prox) %if it's still a string it is invalid!
            errmsg = [' Invalid string supplied for penalty function'... 
                '\n Valid penalty arguments are "l2","snowflake", "log",', ...
                ' or a cell array of {@exact_function, @prox_function} \n'];
            error(errmsg,class(SR3.prox));
        end
    else
        penalty = SR3.prox;
    end
    SR3.rho = penalty{1};
    SR3.proxfun = penalty{2};
    

    if SR3.params.store_updates
        SR3.output.V = {};
        SR3.output.U = {};
    end
    
    %-----SOLVER PARSING-----%
    solver = SR3.solver;
    if ~isstruct(solver)
        error('solver must be a struct')
    end
    if ~isfield(solver,'f')
        solver.f = default.solver.f;
    end
    solver.f = lower(char(solver.f));
    if ~any(strcmp(solver.f,{'pcg','pcg_preconditioned', 'factors','legendre','chebyshev','chebyshev_alt',}))
        fprintf([char("solver.f was invalid. Must be one of") ...
        char("{'pcg','pcg_preconditioned', 'factors','legendre','chebyshev','chebyshev_alt}.")...
        ' Setting to default pcg_preconditioned']);
    end
    if isfield(solver,'params')
        solver.params = default_param_struct(solver.params, default.SR3.solver.params);
    else
        solver.params = default.SR3.solver.params;
    end
    SR3.solver = solver;
end

function A = remap_tensor(a,b)
    %remap a such that it is a tensor the same shape as b was derived from
    A = tensor(tenmat(a,b.rdims,b.cdims,b.tsize));
end
