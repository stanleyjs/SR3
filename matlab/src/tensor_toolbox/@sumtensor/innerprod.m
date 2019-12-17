function res = innerprod(X,Y)
%INNERPROD Efficient inner product with a sumtensor.
%
%   R = INNERPROD(X,Y) efficiently computes the inner product between
%   two tensors X and Y, where X and/or Y are sumtensors.
%
%   See also TENSOR/INNERPROD, SPTENSOR/INNERPROD, TTENSOR/INNERPROD, 
%   KTENSOR/INNERPROD
%
%MATLAB Tensor Toolbox.
%Copyright 2015, Sandia Corporation.

% This is the MATLAB Tensor Toolbox by T. Kolda, B. Bader, and others.
% http://www.sandia.gov/~tgkolda/TensorToolbox.
% Copyright (2015) Sandia Corporation. Under the terms of Contract
% DE-AC04-94AL85000, there is a non-exclusive license for use of this
% work by or on behalf of the U.S. Government. Export of this data may
% require a license from the United States Government.
% The full license terms can be found in the file LICENSE.txt

% X and/or Y is a sumtensor.  We compute the inner product of two sumtensors
% by calling innerprod of Y with each part of X.
res = 0;

for i = 1:length(X.part)
    if isa(Y, 'sumtensor')
        res = res + innerprod(Y, X.part{i});
    else
        res = res + innerprod(X.part{i}, Y);
    end
end



