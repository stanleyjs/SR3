function [psi,lambda] = prodgraph_eigs(phi,k)
    modes = numel(phi);
    for i = 1:modes
        L{i} = phi{i}'*phi{i};
        k_actual = min(k,size(L{i},1));
        [U{i},sig,~] = svds(L{i},k_actual,'smallest');
        S{i} = diag(sig);
    end
    psi = [];
    lambda = [];
    iter = [1:modes];
    for i = 1:modes
        for ii = 1:size(U{i},2)
            for j = i+1:modes
                for jj = 1:size(U{j},2)
                    psi = [psi kron(U{j}(:,jj),U{i}(:,ii))]; 
                    lambda = [lambda; S{i}(ii)+S{j}(jj)];
                end
            end
        end
    end
end