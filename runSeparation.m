function outputFiles = runSeparation(inputFile, mode)
% ========================================================
% SPEAKER SEPARATION PIPELINE
% mode: 'conversation' = VGGish only
%       'simultaneous' = SepFormer only
%       'mixed'        = SepFormer + VGGish mask
% ========================================================

    outputFiles = struct();
    numSpeakers = 2;

    % -------------------- LOAD AUDIO --------------------
    [audioData, sampleRate] = audioread(inputFile);
    disp(['File loaded successfully — mode: ' mode])

    % -------------------- MONO --------------------
    if size(audioData, 2) == 2
        audioData = mean(audioData, 2);
    end

    % -------------------- RESAMPLE TO 16kHz --------------------
    if sampleRate ~= 16000
        audioData = resample(audioData, 16000, sampleRate);
        sampleRate = 16000;
    end

    % ========================================================
    if strcmp(mode, 'simultaneous')
        outputFiles = runSepFormer(audioData, sampleRate, numSpeakers);

    elseif strcmp(mode, 'conversation')
        outputFiles = runVGGish(audioData, sampleRate, numSpeakers);

    elseif strcmp(mode, 'mixed')
        outputFiles = runCombined(audioData, sampleRate, numSpeakers);

    else
        disp('Unknown mode, defaulting to conversation')
        outputFiles = runVGGish(audioData, sampleRate, numSpeakers);
    end

    disp('Separation complete')
    disp(outputFiles)
end


% ========================================================
% PIPELINE 1 — SEPFORMER ONLY (simultaneous speakers)
% ========================================================
function outputFiles = runSepFormer(audioData, sampleRate, numSpeakers)
    outputFiles = struct();
    outputFiles.numSpeakers = numSpeakers;
    outputFiles.isolated = {};
    outputFiles.removed = {};

    disp('Running SepFormer...')
    audio8k = resample(audioData, 8000, sampleRate);
    raw = separateSpeakers(audio8k, 8000, NumSpeakers=numSpeakers);

    % Resample back to 16kHz and match length
    separated = zeros(length(audioData), numSpeakers);
    for spk = 1:numSpeakers
        up = resample(raw(:, spk), sampleRate, 8000);
        if length(up) > length(audioData)
            up = up(1:length(audioData));
        elseif length(up) < length(audioData)
            up = [up; zeros(length(audioData) - length(up), 1)];
        end
        separated(:, spk) = up;
    end

    for spk = 1:numSpeakers
        % Isolated
        iso = separated(:, spk);
        iso = normaliseAudio(iso);
        isoFile = ['speaker_' num2str(spk) '_isolated.wav'];
        audiowrite(isoFile, iso, sampleRate);
        outputFiles.isolated{end+1} = isoFile;

        % Removed = sum of all other speakers
        others = setdiff(1:numSpeakers, spk);
        rem = zeros(length(audioData), 1);
        for o = others
            rem = rem + separated(:, o);
        end
        rem = normaliseAudio(rem);
        remFile = ['speaker_' num2str(spk) '_removed.wav'];
        audiowrite(remFile, rem, sampleRate);
        outputFiles.removed{end+1} = remFile;
    end
    disp('SepFormer complete')
end


% ========================================================
% PIPELINE 2 — VGGISH ONLY (turn-based conversation)
% ========================================================
function outputFiles = runVGGish(audioData, sampleRate, numSpeakers)
    outputFiles = struct();
    outputFiles.numSpeakers = numSpeakers;
    outputFiles.isolated = {};
    outputFiles.removed = {};

    disp('Running VGGish diarization...')

    % VAD
    speechIndices = detectSpeech(audioData, sampleRate);
    speechMask = zeros(length(audioData), 1);
    for i = 1:size(speechIndices, 1)
        speechMask(speechIndices(i,1):speechIndices(i,2)) = 1;
    end

    % Chunk parameters
    chunkSize = round(sampleRate * 1.0);
    hopSize   = round(sampleRate * 0.25);
    numChunks = floor((length(audioData) - chunkSize) / hopSize) + 1;

    % Extract embeddings
    net = vggish;
    embeddings = zeros(numChunks, 128);
    silentChunks = false(numChunks, 1);

    for i = 1:numChunks
        startSample = (i-1) * hopSize + 1;
        endSample   = min(startSample + chunkSize - 1, length(audioData));
        chunk = audioData(startSample:endSample);
        if length(chunk) < chunkSize
            chunk = [chunk; zeros(chunkSize - length(chunk), 1)];
        end
        chunkMask = speechMask(startSample:min(endSample, length(audioData)));
        if mean(chunkMask) < 0.3
            embeddings(i, :) = NaN;
            silentChunks(i) = true;
            continue
        end
        try
            features = vggishPreprocess(chunk, sampleRate);
            embedding = predict(net, features);
            embeddings(i, :) = mean(embedding, 1);
        catch
            embeddings(i, :) = zeros(1, 128);
        end
    end

    embeddings(silentChunks, :) = 0;
    embeddings = (embeddings - mean(embeddings)) ./ (std(embeddings) + 1e-8);

    % K-means
    [sampleLabels, silentSamples] = clusterEmbeddings(embeddings, silentChunks, ...
        numChunks, numSpeakers, hopSize, chunkSize, length(audioData));

    % Build output files
    for spk = 1:numSpeakers
        iso = zeros(length(audioData), 1);
        iso(sampleLabels == spk) = audioData(sampleLabels == spk);
        isoFile = ['speaker_' num2str(spk) '_isolated.wav'];
        audiowrite(isoFile, iso, sampleRate);
        outputFiles.isolated{end+1} = isoFile;

        rem = audioData;
        rem(sampleLabels == spk) = 0;
        remFile = ['speaker_' num2str(spk) '_removed.wav'];
        audiowrite(remFile, rem, sampleRate);
        outputFiles.removed{end+1} = remFile;
    end
    disp('VGGish complete')
end


% ========================================================
% PIPELINE 3 — COMBINED (mixed overlap + turn-based)
% ========================================================
function outputFiles = runCombined(audioData, sampleRate, numSpeakers)
    outputFiles = struct();
    outputFiles.numSpeakers = numSpeakers;
    outputFiles.isolated = {};
    outputFiles.removed = {};

    % --- SepFormer ---
    disp('Combined: Running SepFormer...')
    audio8k = resample(audioData, 8000, sampleRate);
    raw = separateSpeakers(audio8k, 8000, NumSpeakers=numSpeakers);

    separated = zeros(length(audioData), numSpeakers);
    for spk = 1:numSpeakers
        up = resample(raw(:, spk), sampleRate, 8000);
        if length(up) > length(audioData)
            up = up(1:length(audioData));
        elseif length(up) < length(audioData)
            up = [up; zeros(length(audioData) - length(up), 1)];
        end
        separated(:, spk) = up;
    end

    % --- VGGish diarization ---
    disp('Combined: Running VGGish diarization...')
    speechIndices = detectSpeech(audioData, sampleRate);
    speechMask = zeros(length(audioData), 1);
    for i = 1:size(speechIndices, 1)
        speechMask(speechIndices(i,1):speechIndices(i,2)) = 1;
    end

    chunkSize = round(sampleRate * 1.0);
    hopSize   = round(sampleRate * 0.25);
    numChunks = floor((length(audioData) - chunkSize) / hopSize) + 1;

    net = vggish;
    embeddings = zeros(numChunks, 128);
    silentChunks = false(numChunks, 1);

    for i = 1:numChunks
        startSample = (i-1) * hopSize + 1;
        endSample   = min(startSample + chunkSize - 1, length(audioData));
        chunk = audioData(startSample:endSample);
        if length(chunk) < chunkSize
            chunk = [chunk; zeros(chunkSize - length(chunk), 1)];
        end
        chunkMask = speechMask(startSample:min(endSample, length(audioData)));
        if mean(chunkMask) < 0.3
            embeddings(i, :) = NaN;
            silentChunks(i) = true;
            continue
        end
        try
            features = vggishPreprocess(chunk, sampleRate);
            embedding = predict(net, features);
            embeddings(i, :) = mean(embedding, 1);
        catch
            embeddings(i, :) = zeros(1, 128);
        end
    end

    embeddings(silentChunks, :) = 0;
    embeddings = (embeddings - mean(embeddings)) ./ (std(embeddings) + 1e-8);

    [sampleLabels, silentSamples, confidence] = clusterEmbeddings(embeddings, ...
        silentChunks, numChunks, numSpeakers, hopSize, chunkSize, length(audioData));

    % --- Apply diarization mask to SepFormer output ---
    disp('Combined: Applying diarization mask...')
    confidenceThreshold = 0.75;
    fadeLen = round(sampleRate * 0.05);

    for spk = 1:numSpeakers
        others = setdiff(1:numSpeakers, spk);
        iso = separated(:, spk);

        mask = ones(length(audioData), 1);
        for other = others
            otherRegion = (sampleLabels == other) & ...
                          (confidence >= confidenceThreshold) & ...
                          ~silentSamples;
            mask(otherRegion) = 0.1;
        end

        mask = smoothMask(mask, fadeLen);
        iso = iso .* mask;
        iso = normaliseAudio(iso);

        isoFile = ['speaker_' num2str(spk) '_isolated.wav'];
        audiowrite(isoFile, iso, sampleRate);
        outputFiles.isolated{end+1} = isoFile;

        rem = zeros(length(audioData), 1);
        for other = others
            rem = rem + separated(:, other);
        end
        rem = normaliseAudio(rem);
        remFile = ['speaker_' num2str(spk) '_removed.wav'];
        audiowrite(remFile, rem, sampleRate);
        outputFiles.removed{end+1} = remFile;
    end
    disp('Combined complete')
end


% ========================================================
% SHARED HELPER — K-means clustering + voting
% ========================================================
function [sampleLabels, silentSamples, confidence] = clusterEmbeddings( ...
        embeddings, silentChunks, numChunks, numSpeakers, hopSize, chunkSize, audioLen)

    bestLabels  = [];
    bestInertia = inf;
    tempDistances = zeros(size(embeddings, 1), numSpeakers);

    for attempt = 1:10
        randIdx = randperm(size(embeddings, 1), numSpeakers);
        tempCentroids = embeddings(randIdx, :);
        tempLabels = zeros(size(embeddings, 1), 1);

        for iter = 1:100
            for k = 1:numSpeakers
                d = embeddings - tempCentroids(k, :);
                tempDistances(:, k) = sum(d .^ 2, 2);
            end
            [~, tempLabels] = min(tempDistances, [], 2);
            for k = 1:numSpeakers
                if any(tempLabels == k)
                    tempCentroids(k, :) = mean(embeddings(tempLabels == k, :));
                end
            end
        end

        [minDists, ~] = min(tempDistances, [], 2);
        attemptInertia = sum(minDists);
        if attemptInertia < bestInertia
            bestInertia = attemptInertia;
            bestLabels  = tempLabels;
        end
    end

    chunkLabels = bestLabels;
    chunkLabels(silentChunks) = 0;

    % Smooth
    smoothedLabels = chunkLabels;
    windowSize = 5;
    for i = windowSize:length(chunkLabels)
        window = chunkLabels(i-windowSize+1:i);
        if chunkLabels(i) ~= 0
            nonSilent = window(window ~= 0);
            if ~isempty(nonSilent)
                smoothedLabels(i) = mode(nonSilent);
            end
        end
    end

    % Voting matrix
    votingMatrix = zeros(audioLen, numSpeakers);
    for i = 1:numChunks
        if smoothedLabels(i) == 0; continue; end
        startSample = (i-1) * hopSize + 1;
        endSample   = min(startSample + chunkSize - 1, audioLen);
        votingMatrix(startSample:endSample, smoothedLabels(i)) = ...
            votingMatrix(startSample:endSample, smoothedLabels(i)) + 1;
    end

    [maxVotes, sampleLabels] = max(votingMatrix, [], 2);
    totalVotes   = sum(votingMatrix, 2);
    silentSamples = totalVotes == 0;
    sampleLabels(silentSamples) = 0;

    confidence = zeros(audioLen, 1);
    hasVotes = totalVotes > 0;
    confidence(hasVotes) = maxVotes(hasVotes) ./ totalVotes(hasVotes);
end


% ========================================================
% SHARED HELPER — normalise audio to 0.95 peak
% ========================================================
function audio = normaliseAudio(audio)
    maxVal = max(abs(audio));
    if maxVal > 0
        audio = audio / maxVal * 0.95;
    end
end


% ========================================================
% SHARED HELPER — smooth mask edges to avoid clicks
% ========================================================
function smoothed = smoothMask(mask, fadeLen)
    smoothed = mask;
    n = length(mask);
    i = 1;
    while i <= n
        if mask(i) < 0.5
            j = i;
            while j <= n && mask(j) < 0.5
                j = j + 1;
            end
            fadeInEnd = min(i + fadeLen - 1, j - 1);
            for k = i:fadeInEnd
                smoothed(k) = mask(k) * (k - i) / fadeLen;
            end
            fadeOutStart = max(j - fadeLen, i);
            for k = fadeOutStart:j-1
                smoothed(k) = mask(k) * (j - k) / fadeLen;
            end
            i = j;
        else
            i = i + 1;
        end
    end
end