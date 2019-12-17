%DEFAULT_PARAM_STRUCT
% Given a struct of params and a struct of defaults, fill in the missing
% fields of params with the corresponding defaults.
function params = default_param_struct(params,defaults)
    assert(isstruct(params), 'Input parameters must be a struct');
    assert(isstruct(defaults), 'Input defaults must be a struct');

    keys = fieldnames(defaults);
    for k=1:numel(keys)
        if ~(isfield(params,keys{k}))
            params.(keys{k}) = defaults.(keys{k});
        end
    end
end