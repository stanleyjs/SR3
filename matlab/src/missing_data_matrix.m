function [X,X_orig,mask,row_labels,col_labels,gamma_vec,kNN,origPts] = ...
    missing_data_matrix(dataset,params,ppp)
if nargin < 2
    params = [];
end
if (~isfield(params,'doPerm'))||isempty(params.doPerm)
    doPerm = false;
else
    doPerm = params.doPerm;
end
if (~isfield(params,'addNoise') || isempty(params.addNoise))
    addNoise = false;
    noise_sig = 0;
else
    addNoise = params.addNoise;
    if ~addNoise
        noise_sig = 0;
    else
        if (~isfield(params,'noise_sig')||isempty(params.noise_sig))
            noise_sig = 0.5;
        else
            noise_sig = params.noise_sig;
        end
    end
end
%%
if ismac
    addpath('/Users/mishne/Documents/redsvn/code/Questionnaire/');
elseif isunix
    addpath('/data/Gal/Questionnaire/');
end
%% load data
if ~exist('dataset','var')
    dataset = 'lung100';
end
if strcmp(dataset,'checker') && ~addNoise
    noise_sig = 0.1;
    addNoise = true;
end
[X,gamma_vec,kNN,col_labels,row_labels,origPts] = load_data(dataset);
[n_rows, n_cols] = size(X);
%% manipulate data
X_orig = X;
if addNoise
    X = X + noise_sig*rand(size(X));
end
if doPerm
    row_perm = randperm(n_rows);
    col_perm = randperm(n_cols);
    X = X(row_perm,col_perm);
else
    row_perm = 1:n_rows;
    col_perm = 1:n_cols;
end
if strcmp(dataset,'mmpi')
    col_labels = scores(1,col_perm);
elseif ~isempty(col_labels)
    col_labels = col_labels(col_perm);
else
    col_labels = col_perm;
end
if ~isempty(row_labels)
    row_labels = row_labels(row_perm);
end
%% remove values
if ~exist('ppp','var')
    ppp = 0.5;
end
    
mask           = true( size( X ) );
zerosNum       = ppp;
permVec        = randperm( n_rows * n_cols, round(  n_rows * n_cols * zerosNum) );
mask( permVec) = false;
X              = mask .* X;