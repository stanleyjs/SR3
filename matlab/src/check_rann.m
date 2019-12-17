function err = check_rann(pth)
    if nargin == 0 || (~ischar(pth) && ~isstring(pth)) || inpath(pth)
        pth = false;
    end
    try
        assert(exist('rann32c')>0)
        nargin('rann32c');
        err = true;
    catch exception
        if pth
            addpath(char(pth));
            nargin('rann32c');
            check_rann()
        else
            err = false;
            id = exception.identifier;
            msg = exception.message;
            msg = [msg '\nVerify that function RANN32C is compiled'];
            switch id
                case 'MATLAB:UndefinedFunction'
                    msg = [msg ' and in the PATH.'];
                case 'MATLAB:scriptNotAFunction'
                    msg = [msg ' and in the PATH with no conflicting scripts.'];
                otherwise
                    rethrow(exception);
            end
            exception = MException(id, msg);
            throw(exception);
        end
    end
end