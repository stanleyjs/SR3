function [AdTAd,Ad,lmax] = tensor_incidence(phi,compute_lmax)
    %build sum_d A^T_d A_d using the mixed product property 
    %needs to be reoptimized...
    lmax = 0;
    D = numel(phi);
    ds = cellfun(@(x) size(x,2),phi);
    rem_d = prod(ds(1:end));
    AdTAd = sparse(rem_d,rem_d);
    AdT = cell(D,1);
    rem_d = rem_d/ds(1);
    phid = phi{1};
    Ad{1} = kron(speye(rem_d),phid);
    phiTphi = phid'*phid;
    if compute_lmax
        lmax = lmax + svds(phiTphi,1);
    end
    AdTAd = AdTAd+kron(speye(rem_d),phiTphi);
    cur_d = 1;
    for ix = 2:D
        rem_d = rem_d/ds(ix);
        cur_d = cur_d*ds(ix-1);
        phid = phi{ix};
        if compute_lmax
            lmax = lmax+svds(phid'*phid,1);
        end
        Ad{ix} =kron(speye(rem_d), phid);
        AdTAd = AdTAd+kron(Ad{ix}'*Ad{ix}, speye(cur_d));
        Ad{ix} =  kron(Ad{ix}, speye(cur_d));
    end
end