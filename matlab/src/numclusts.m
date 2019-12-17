function nc = numclusts(dists)
    nc = 0;
    to_check = ones(size(dists,1),1);
    while any(to_check)
        i = find(to_check,1);
        j = dists(i,:)<1e-15;
        nc = nc+1;
        to_check(j) = 0;
        to_check(i) = 0;
    end
end