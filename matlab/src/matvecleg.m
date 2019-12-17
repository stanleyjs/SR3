function y = matvecleg(M, v, order, coeffs)
    %% compute the order-th legendre polynomial approximaton of f(M)v
    %% where f is described by coeffs 
    %%
    %% The eigenvalues of M must be warped to [-1,1]
    %%
    %% The reccurrence is 
    %% Pk+1(M)*v = ((2*k+1) * M * Pk(M) - k * Pk-1(M))/(k+1) * v;
    %% = ((2*k+1) * M * (Pk(M)*v) - k * Pk-1(M)*v)/(k+1)
    %% the normalization for Pk is sqrt((2*k+1)/2)
    P0v = eye(size(M,1));
    P0v = P0v*v;
    P1v = M;
    P1v = P1v*v;
    y = coeffs(1)*P0v+coeffs(2)*P1v;
    
    for k = 2:(order-1)
        P2v = ((2*k-1)*M * P1v - (k-1)*P0v)/(k);
        y = y + coeffs(k+1)*P2v;
        P0v = P1v;
        P1v = P2v;  
    end

end