function Pk = nextlegendre(Pkm1, Pkm2, x, k)

Pk = @(x) ((2*k+1) * x * Pkm1(x) - k * Pkm2(x))/(k+1);

end