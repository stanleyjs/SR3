function x_smooth = calculate_averaging_matrix(x,n_clust, cc)

x_smooth = tensor(x);

for idim = 1:ndims(x)
    avg_mat = zeros(size(x,idim),size(x,idim));
    for cind = 1:n_clust(idim)
        c_inds = find(cc{idim} == cind);
        avg_mat(c_inds,c_inds) = 1 / length(c_inds);
    end
    x_smooth = tenmat(x_smooth, idim);
    x_smooth(:) = avg_mat * x_smooth;
    x_smooth = tensor(x_smooth);
end

x_smooth = double(tensor(x_smooth));
end