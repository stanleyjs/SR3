function out = l2_prox(v,sigma,epsilon)
    tmp2 = sigma'  ./sqrt(sum(v.^2,1));
    tmp  = 1 - tmp2;
    out = bsxfun(@times,v , max(0,tmp));
end