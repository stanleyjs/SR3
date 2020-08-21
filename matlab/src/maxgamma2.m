function [max_gamma_vec, min_gamma_vec] = maxgamma2(x,phi,SR3,min_gamma)

max_gamma_vec = zeros(1, ndims(x));
min_gamma_vec = zeros(1, ndims(x)); 

parfor idim = 1:ndims(x)
    n_clust = size(x,idim);
    gammas = zeros(1, ndims(x));
    gammas(idim) = min_gamma;
    %this_SR3 = SR3;
    while n_clust > 1
        fprintf('gamma_%d = %.4f\n',  idim, gammas(idim));
        [results] = SR3_tensor(x,phi,SR3,gammas);
        %this_SR3.U0 = results.output.U;
        V = results.output.V;
        %this_SR3.V0 = V;
        mr=(bsxfun(@times,~vecnorm(double(V{idim}),2,2),double(phi{idim, 1})));
        L = mr'*mr;
        G = graph(L);
        
        cc = conncomp(G);
        n_clust = max(cc);
        
        if n_clust < size(x,idim) && min_gamma_vec(idim) == 0
            min_gamma_vec(idim) =  gammas(idim);
        end
        
        gammas(idim) =  gammas(idim) * 2;
    end
    max_gamma_vec(idim) = gammas(idim) / 2;
end
        
end        

%%%
% test
% [results] = SR3_tensor(x,phi,SR3_2,[0 max_gamma(2)]);
% kk=1;
% U{kk} = results.output.U;
% V{kk} = results.output.V;
% F{kk} = results.output.F;
% mr=(bsxfun(@times,~vecnorm(double(V{kk}{1}),2,2),double(phi{1, 1})));
% Lr = mr'*mr;
% G_r = graph(Lr);
% mc=(bsxfun(@times,~vecnorm(double(V{kk}{2}),2,2),double(phi{2, 1})));
% Lc = mc'*mc;
% G_c = graph(Lc);
% cc_rows = conncomp(G_r);
% cc_labels_rows{kk} = cc_rows;
% cc_cols = conncomp(G_c);
% cc_labels_cols{kk} = cc_cols;
% nP_c = max(cc_cols)
% nP_r = max(cc_rows)