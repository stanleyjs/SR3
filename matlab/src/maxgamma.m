function gamma = maxgamma(L,A,phi,x,nu,epsilon)
    mat = (L+nu*speye(size(L,1)));
    xvec =reshape(x,[],1);
    ustar = pcg(mat,nu*xvec,1e-8,100);
    
    gamma = zeros(1,ndims(x));
    for i = 1:ndims(x)
        uij = A{i}*ustar;
        stride = size(phi{i},1);
        otherdim = max(size(uij))/stride;
        uij = reshape(tensor(uij), [stride,otherdim]);
        temp = vecnorm(double(uij)');
        gamma(i) = max((2*epsilon)/nu .* temp);
    end
end