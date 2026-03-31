% This is on its way to being deprecated since the functionality of this
% method is also done in the Dataset class inside the modules folder.

function [spike_class,spikeVecs,spikeCounts,spikeIndices] = readData(path,baseName,vectorSize,startOffset,noiseLevel)
% Read spikes from raw data with specified offset and window size
%   Detailed explanation goes here
    numClasses = 3;

    dataFile = strcat(path,baseName,"_data.txt");
    classFile = strcat(path,baseName,"_spikeClass.txt");
    timeFile = strcat(path,baseName,"_spikeTimes.txt");
    
    % Load data from files
    f = fopen(dataFile,'r');
    data = fscanf(f,"%f");
    fclose(f);
    
    f = fopen(classFile,'r');
    spike_class = fscanf(f,"%f");
    fclose(f);
    
    f = fopen(timeFile,'r');
    spike_times = fscanf(f,"%f");
    fclose(f);
    
    %
    %Get Spikes (bypass detecting)
    %  
    numSpikes = length(spike_times);
    spikeVecs = zeros([numSpikes,vectorSize]);
    spikeCounts = zeros(1,numClasses);
    spikeIndices = zeros(numClasses,1500);
    for spikeIdx=1:numSpikes
        start = spike_times(spikeIdx)+startOffset;
        for pointIdx=1:vectorSize
            spikeVecs(spikeIdx,pointIdx) = data(start+pointIdx-1);
        end

        class = spike_class(spikeIdx);
        spikeCounts(class) = spikeCounts(class)+1;
        spikeIndices(class,spikeCounts(class)) = spikeIdx;

        if noiseLevel ~= 0
            % noise_std = noiseLevel*max(abs(spikeVecs(spikeIdx,:)));
            % noisy = noise_std*randn(1,vectorSize);
            % spikeVecs(spikeIdx,:) = spikeVecs(spikeIdx,:)+noisy;
        
            noisy_sig = awgn(spikeVecs(spikeIdx,:),20*log10(1/noiseLevel),'measured');
            spikeVecs(spikeIdx,:) = noisy_sig;
        end

    end
    
end