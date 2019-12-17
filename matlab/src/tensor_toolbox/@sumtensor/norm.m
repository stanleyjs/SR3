function n = norm(T)
%NORM Frobenius norm of a sumtensor.
%
%   It is not possible to efficiently compute the NORM of sumtensor. 
%   We print a warning and use an inefficient inner product. 
%   If this fails we return 0. This function is included for compatibility
%   with certain routines that expect to compute
%   the norm but don't *really* need it.
%   
%   NORM(X) returns 0.
%
%   See also SUMTENSOR.
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


warning(['An efficient implementation of the NORM function for SUMTENSOR' ... 
    ' is not available. \n Attempting INNERPROD implementation \n']);
try
    n = sqrt(innerprod(T,T));
catch e
    warning(['INNERPROD failed with error: \n%s' e.message ',' ...
        ' \n returning zero.']);
    n = 0;
end