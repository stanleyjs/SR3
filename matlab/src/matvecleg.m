function y = matvecleg(M1,M2, v, order, coeffs)
    %% compute the order-th legendre polynomial approximaton of f(M)v
    %% where f is described by coeffs 
    %%
    %% The eigenvalues of M must be warped to [-1,1]
    %%
    %% The reccurrence is 
    %% Pk+1(M)*v = ((2*k+1) * M * Pk(M) - k * Pk-1(M))/(k+1) * v;
    %% = ((2*k+1) * M * (Pk(M)*v) - k * Pk-1(M)*v)/(k+1)
    %% the normalization for Pk is sqrt((2*k+1)/2)
    n = size(M1,1);
    pk = v;
    pkp1 =M2*(M1'*v);
    y = coeffs(1)*pk+coeffs(2)*pkp1;

    for k = 2:(order-1)
        pkm1=pk;
        pk= pkp1;
        pkp1=( (2*k-1)*M2*(M1'*pk)-(k-1)*pkm1)/(k);
        y = y+coeffs(k+1)*pkp1;
    end

end