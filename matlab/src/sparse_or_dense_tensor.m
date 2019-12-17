% SPARSE_OR_DENSE_TENSOR
% cast X as a tensor toolbox tensor of the correct density
% sp is true if X is an sptensor.
function [X,sp] = sparse_or_dense_tensor(X,force_sparse,pth)
    
    if nargin == 1
        force_sparse = false;
    end
    if ~exist('pth') || (~ischar(pth) && ~isstring(pth)) || inpath(pth)
        pth = false;
    end
    if ~contains(class(X),'tensor')
        try
            if nnz(X)<0.5*numel(X) || force_sparse
                X = sptensor(X);
                sp = true;
            else
                X = tensor(X);
                sp = false;
            end
        catch exception
            if pth
                addpath(char(pth));
                [X,sp] = sparse_or_dense_tensor(X,force_sparse);
            else
                id = exception.identifier;
                msg = exception.message;
                msg = [msg '\nVerify that the tensor toolbox is installed'];
                switch id
                    case 'MATLAB:UndefinedFunction'
                        msg = [msg ' and in the PATH.'];
                    case 'MATLAB:scriptNotAFunction'
                        msg = [msg ' and in the PATH with no conflicting scripts.'];
                    otherwise
                        rethrow(exception)
                end
                exception = MException(id, msg);
                throw(exception);
            end
        end
    else
        if nnz(X)<0.5*numel(X) || force_sparse
            X = sptensor(X);
            sp = true;
        else
            X = tensor(X);
            sp = false;
        end
    end
end
