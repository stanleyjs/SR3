function A = remap_tensor(a,b)
    %remap a such that it is a tensor the same shape as b was derived from
    A = tensor(tenmat(a,b.rdims,b.cdims,b.tsize));
end
