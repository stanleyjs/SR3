function kwargs = struct2kwargs(structure)
    keys = fieldnames(structure);
    vals = struct2cell(structure);
    for k=1:numel(keys)
        kwargs(:,k) = {keys{k}; vals{k}};
    end
end