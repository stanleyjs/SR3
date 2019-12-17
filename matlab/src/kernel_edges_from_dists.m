function Phi = kernel_edges_from_dists(dists,params)
        if nargin == 1
            params = struct();
        end
        defaults.k = 5; % kNN
        defaults.thresh = 0.1;
        defaults.connect = 0;
        defaults.nosingleton = 1;
        defaults.bandwidthfun = @(x) min(mean(x));
        params = default_param_struct(params,defaults);
        n = size(dists,1);

        [knndists,ix] = sort(dists,2);
        kdists = knndists(:,params.k+1);
        kdists = params.bandwidthfun(kdists);
        kernel = exp(-(dists.^2)./(2*kdists.^2));
        kernel = kernel - diag(diag(kernel));
        kernel = kernel >= params.thresh;
        if params.nosingleton
            if any(~any(kernel,1))
                fprintf('Disconnected point detected, connecting to closest neighbor');
            end
            
            kernel(find(~any(kernel),1),ix(~any(kernel,1),2)) = 1;
            kernel(ix(~any(kernel,1),2),find(~any(kernel,1))) = 1;
        end
        nnz(tril(kernel))

        [I,J] = find(tril(kernel));
        edges =[I J];
        if params.connect %% clean up connected components by adding random edges from each
            %% component
            g = graph(I,J);
            CC = conncomp(g);
            for c = 1:max(CC)
                ctmp = randperm(length(CC));
                cix = find(CC(ctmp)==c,1); %a random element from component i%
                for j = (c+1):max(CC)
                    jix = find(CC(ctmp)==j,1);
                    edges = [edges; ctmp(cix)' ctmp(jix)'];
                end
            end
        end
        nE = size(edges,1);
        Phi = sparse(repmat([1:nE]',2,1),[edges(:,1); edges(:,2)],...
            [ones(nE,1);-1*ones(nE,1)],nE,n);

end