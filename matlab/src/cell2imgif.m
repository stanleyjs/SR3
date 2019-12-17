function fig = cell2imgif(cellmat,filename, overwrite, delaytime, skip,showframe, cscale)
% given a cell of matrices `cellmat`, write a gif to `filepath.gif` with frame delay
%`delaytime` and frame skip `skip`
if nargin<=6
    cscale = true;
    if nargin<=5
        showframe = false;
        if nargin<=4

            skip = 1;
            if nargin <=3
                delaytime = 0.5;
                if nargin <=2
                    overwrite = false;
                    if nargin==1
                         filename = 'cell2imgif_output.gif';
                    end
                end
            end
        end
    end
end
    if ~contains(filename,'.gif')
        filename  = [filename '.gif'];
    end
    file_to_write = filename;
    if ~overwrite
        i = 0;
        while isfile(file_to_write)
            i = i+1;
            file_to_write = insertBefore(filename,'.gif', ['(' num2str(i) ')']);
        end
    end
        
        
        
    fig=figure('visible', 'off');
    axis tight;
    maxval = max(cellfun(@(x)max(max(double(x))),cellmat));
    minval = min(cellfun(@(x)min(min(double(x))),cellmat));

    for i=1:skip:numel(cellmat)
        t = double(cellmat{i});
        imagesc(t)
        colormap bone;
        if cscale
            caxis([minval,maxval])
        end
        if showframe
            title(['Iteration ' num2str(i)])
        end
        drawnow
        frame = getframe(gcf); 
          im = frame2im(frame); 
          [imind,cm] = rgb2ind(im,256); 
          % Write to the GIF File 
          if i == 1 
              imwrite(imind,cm,file_to_write,'gif', 'Loopcount',inf,'DelayTime',delaytime); 
          else 
              imwrite(imind,cm,file_to_write,'gif','WriteMode','append','DelayTime', delaytime); 
          end 
    end
    close gcf
end