% ========================================================
% MATLAB TCP SERVER FOR SPEAKER SEPARATION
% Run by typing: startServer
% ========================================================
disp('========================================')
disp('Speaker Separation Server')
disp('Open browser and go to: http://localhost:8080')
disp('Press Ctrl+C to stop the server')
disp('========================================')

tcpServer = tcpserver("0.0.0.0", 8080);
disp('Server started, waiting for connections...')

while true
    try
        if tcpServer.NumBytesAvailable > 0
            pause(0.05)
            rawData = read(tcpServer, tcpServer.NumBytesAvailable, 'uint8');
            request = char(rawData);
            lines = strsplit(request, newline);
            firstLine = strtrim(lines{1});
            disp(['Request: ' firstLine])

            if contains(firstLine, 'POST /process')
                disp('Audio received, processing...')
                handleProcessing(tcpServer, request, rawData);

            elseif contains(firstLine, 'GET /download')
                parts = strsplit(firstLine, ' ');
                uri = parts{2};
                filename = uri(11:end);
                serveAudioFile(tcpServer, filename);

            elseif contains(firstLine, 'GET /')
                serveFile(tcpServer, 'index.html', 'text/html');

            else
                send404(tcpServer);
            end
        end
    catch err
        disp(['Error: ' err.message])
    end
    pause(0.05)
end

% ========================================================
function serveFile(tcpServer, filename, contentType)
    try
        fid = fopen(filename, 'rb');
        contentBytes = fread(fid, '*uint8');
        fclose(fid);
        contentBytes = contentBytes(:);
        header = sprintf('HTTP/1.1 200 OK\r\nContent-Type: %s; charset=utf-8\r\nContent-Length: %d\r\nAccess-Control-Allow-Origin: *\r\n\r\n', ...
            contentType, length(contentBytes));
        write(tcpServer, [uint8(header(:)); contentBytes]);
        disp(['Served: ' filename])
    catch err
        disp(['Error serving file: ' err.message])
        send404(tcpServer);
    end
end

function serveAudioFile(tcpServer, filename)
    try
        fid = fopen(filename, 'rb');
        audioBytes = fread(fid, '*uint8');
        fclose(fid);
        audioBytes = audioBytes(:);
        header = sprintf('HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nContent-Length: %d\r\nAccess-Control-Allow-Origin: *\r\nContent-Disposition: attachment; filename="%s"\r\n\r\n', ...
            length(audioBytes), filename);
        write(tcpServer, [uint8(header(:)); audioBytes]);
        disp(['Served audio: ' filename])
    catch err
        disp(['Error serving audio: ' err.message])
        send404(tcpServer);
    end
end

function handleProcessing(tcpServer, request, rawData)
    try
        % Extract mode from headers
        mode = 'conversation';
        headerLines = strsplit(request, newline);
        for i = 1:length(headerLines)
            if contains(headerLines{i}, 'X-Separation-Mode:')
                parts = strsplit(headerLines{i}, ':');
                mode = strtrim(parts{2});
                break
            end
        end
        disp(['Mode selected: ' mode])

        % Extract audio bytes
        headerEnd = strfind(request, [char(13) char(10) char(13) char(10)]);
        if isempty(headerEnd)
            headerEnd = strfind(request, [char(10) char(10)]);
            offset = 2;
        else
            offset = 4;
        end

        if isempty(headerEnd)
            sendError(tcpServer, 'Could not parse request');
            return
        end

        audioBytes = rawData(headerEnd(1) + offset : end);
        fid = fopen('uploaded_audio.wav', 'wb');
        fwrite(fid, audioBytes, 'uint8');
        fclose(fid);

        disp('Running separation pipeline...')
        outputFiles = runSeparation('uploaded_audio.wav', mode);

        jsonResp = jsonencode(outputFiles);
        header = sprintf('HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\nAccess-Control-Allow-Origin: *\r\n\r\n', ...
            length(jsonResp));
        write(tcpServer, [uint8(header(:)); uint8(jsonResp(:))]);
        disp('Response sent!')

    catch err
        disp(['Processing error: ' err.message])
        sendError(tcpServer, err.message);
    end
end

function sendError(tcpServer, message)
    body = ['{"error": "' message '"}'];
    header = sprintf('HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: %d\r\nAccess-Control-Allow-Origin: *\r\n\r\n', ...
        length(body));
    write(tcpServer, [uint8(header(:)); uint8(body(:))]);
end

function send404(tcpServer)
    body = '404 Not Found';
    header = sprintf('HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: %d\r\n\r\n', ...
        length(body));
    write(tcpServer, [uint8(header(:)); uint8(body(:))]);
end