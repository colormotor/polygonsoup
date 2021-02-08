% Matlab interface to the axidraw_server.py script
function axi(cmd, data, address, port)
    precision = 4;
    
    if nargin < 4
        port = 80;
    end
    if nargin < 3
        address = '127.0.0.1';
    end

    t = tcpip(address, port);
    t.OutputBufferSize = 1500000; 
    fopen(t);
    
    try
        if strcmp(cmd,'title')
            str = sprintf('PATHCMD title %s', data);
            fprintf(t, str);
        elseif strcmp(cmd,'pen')
            if data > 0
                fprintf(t, 'PATHCMD pen_down');
            else
                fprintf(t, 'PATHCMD pen_up');
            end
        elseif strcmp(cmd,'draw') || strcmp(cmd,'draw_raw')
            display('drawing');
            fprintf(t, 'PATHCMD drawing_start');
            if iscell(data)
                for i=1:size(data,2)
                    send_path(t, data{i}, precision);
                end
            else
                send_path(t, data, precision);
            end
            if strcmp(cmd,'draw')
                fprintf(t, 'PATHCMD drawing_end');
            else
                fprintf(t, 'PATHCMD drawing_end_raw');
            end
        else
        end 
    catch ME
        getReport(ME)
        display('error, closing connection\n');
    end
    
    echotcpip('off')
    fclose(t);
    delete(t);
    clear t;
    
end

function send_path(t, P, precision)
    P = P(1:2,:); %make sure it is 2d
    n = size(P,2);
    x = reshape(P, 1, n*2);
    str = mat2str(x, precision);
    str = str(2:end-1); % remove trailing brackets
    str = sprintf('PATHCMD stroke %d %s', n, str);
    fprintf(t, str);
end