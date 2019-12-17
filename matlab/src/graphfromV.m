function Ahat = graphfromV(V,phi)
    dim = size(V);
    Vhat = sum(V,2);
    gate = Vhat~=0;
    gate = ~gate;
    phi = phi(gate,:);
    Ahat = kron(eye(dim(2)),phi);
end