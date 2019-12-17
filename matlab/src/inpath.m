function tf = inpath(pth)
    pathCell = regexp(path, pathsep, 'split');
    if ispc  % Windows is not case-sensitive
      tf = any(strcmpi(pth, pathCell));
    else
      tf = any(strcmp(pth, pathCell));
    end
end