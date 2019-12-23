function y = legpoint(x, coeffs)
    %% Pk+1(M)*v = ((2*k+1) * M * Pk(M) - k * Pk-1(M))/(k+1) * v;
    %% = ((2*k+1) * M * (Pk(M)*v) - k * Pk-1(M)*v)/(k+1)
    n = numel(coeffs);
    pk = 1;
    pkp1 = x;
    y = coeffs(1)*pk + coeffs(2)*pkp1; 
    for k = 2:(n-1)
        pkm1=pk;
        pk= pkp1;
        pkp1=( (2*k-1)*x*pk-(k-1)*pkm1 )/(k);
        y = y+coeffs(k+1)*pkp1;
    end

   
end