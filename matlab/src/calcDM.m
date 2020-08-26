function diffMaps = calcDM(X,params)
dist_mat = squareform(pdist(X));
eps      =  median(dist_mat(:));
aff_mat_row = exp(-dist_mat.^2 / eps.^2);
[ vecs, vals ] = CalcEigs( aff_mat_row, params.nEigs );
diffMaps = vecs*vals;