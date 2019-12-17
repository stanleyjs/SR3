function Phi = graph_from_knn(idx,params)
        if nargin == 1
            params = struct();
        end
        defaults.k = 5; % kNN
        defaults.connect = 0;
        if isfield(params,'knnparams')
            if isfield(params.knnparams,'k')
                params.k = params.knnparams.k;
            end
        end
        params = default_param_struct(params,defaults);
        n = size(idx,1);
        I = repmat([1:n]',params.k,1);
        J = reshape(idx,[],1); %matlab does column matricizaiton
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
        [~,kx] = unique(sort(edges,2),'rows','stable'); %the knn should be nearly symmetric
        edges = edges(kx,:);
        
        edges(edges(:,1) == edges(:,2),:) = [];
        nE = size(edges,1);
        Phi = sparse(repmat([1:nE]',2,1),[edges(:,1); edges(:,2)],...
            [ones(nE,1);-1*ones(nE,1)],nE,n);
end