function [X, phi] = blockdiag_example(varargin)
    p = inputParser;
    default.Pc = 4;
    default.Pr = 2;
    default.N = 20;
    default.M = 20;
    default.mu = 0;
    default.sigma = 0.001;
    addOptional(p, 'N',default.N, @(x) isnumeric(x));
    addOptional(p, 'M', default.M, @(x) isnumeric(x));
    addOptional(p, 'Pc', default.Pc, @(x) isnumeric(x));
    addOptional(p, 'Pr', default.Pr, @(x) isnumeric(x));

    addOptional(p, 'mu', default.mu, @(x) isnumeric(x));
    addOptional(p, 'sigma',default.sigma, @(x) isnumeric(x));
    parse(p,varargin{:});
    
    args = p.Results;
    N = args.N;
    Pc = args.Pc;
    Pr = args.Pr;
    M = args.M;
    mu = args.mu;
    sigma = args.sigma;
    X = zeros(M,N);

    rem = mod(N,Pc);
    stride = floor(N/Pc);
    column_identities = ones(1,N);
    for i=1:(Pc-1)
        column_identities(i*stride+1:(i+1)*stride) = i+1;
    end
    column_identities(end-rem:end) = i+1;
    muC = mu.*[1:Pc];
    for i = 1:Pc
        mask = column_identities==i;
        X(:,mask) = repmat(muC(i),M,sum(mask))+randn(M,sum(mask)).*sigma;
    end


    if Pr>1
        rem = mod(M,Pr);
        stride = floor(M/Pr);
        row_identities = ones(1,M);
        for i=1:(Pr-1)
            row_identities(i*stride+1:(i+1)*stride) = i+1;
        end
        row_identities(end-rem:end) = i+1;
        muR = mu.*[1:Pr];
        for i = 1:Pr
            mask = row_identities==i;
            X(mask,:) = X(mask,:)+(repmat(muR(i),sum(mask),N)+randn(sum(mask),N).*sigma);
        end
        row_edges = [];
        for i = 1:M-1
            row_edges = [row_edges; i i+1];
        end
        nE = size(row_edges,1);
        phi{1} = sparse(repmat([1:nE]',2,1),[row_edges(:,1); row_edges(:,2)],...
            [ones(nE,1);-1*ones(nE,1)],nE,M);
    end

%     
%     if isscalar(M)
%     else
%         X = zeros(sum(M),N);
%         row_identities = ones
%     end
%     
%     if isscalar(mu)
%         mu = repmat(mu.*[1:P],M,1);
%     else
%         if all(size(mu)==[P M])
%             mu = mu';
%         end
%         assert(all(size(mu) == [M P]))
%     end
%     
%     if isscalar(sigma)
%         temp = repmat(sigma.*eye(M), 1, P);
%         sigma = temp;
%     else
%         if all(size(sigma)==[P M]) || all(size(sigma) == [M*P M])
%             sigma = sigma';
%         end
%         if all(size(sigma) == [M M*P])
%             true;
%         else
%             temp = zeros(M,M*P);
%             for i = 1:P
%                 temp(:,i:i*M) = diag(sigma(:,i));
%             end
%         end
%     end
%     
%     for i = 1:P
%         mask = column_identities==i;

    column_edges = [];
    for i = 1:N-1
        column_edges = [column_edges; i i+1];
    end
    nE = size(column_edges,1);
    phi{2} = sparse(repmat([1:nE]',2,1),[column_edges(:,1); column_edges(:,2)],...
        [ones(nE,1);-1*ones(nE,1)],nE,N);


end