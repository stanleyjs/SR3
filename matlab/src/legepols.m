function y = legepols(x, n)
    %% compute the n-1-th legendre polynomials  of f(x)
    %%
    %%
    %% The reccurrence is 
    %% Pk+1(M)*v = ((2*k+1) * x * Pk(x) - k * Pk-1(x))/(k+1)
    y = zeros(n,numel(x));
%     x = fliplr(x);
    y(1,:) = 1;
    y(2,:) = x;
    for k = 2:(n-1)
        tmp = legendre(k,x);
        y(k+1,:) = tmp(1,:);
    end

end