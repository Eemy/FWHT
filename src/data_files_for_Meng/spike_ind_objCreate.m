path = "../../../SpikeTraining_RISCV/Raw_Data/";

%% Set up parameters for run
baseName = ["C_Easy1_noise005","C_Easy1_noise01","C_Easy1_noise015","C_Easy1_noise02", ...
            "C_Easy1_noise025","C_Easy1_noise03","C_Easy1_noise035","C_Easy1_noise04", ...
            "C_Easy2_noise005","C_Easy2_noise01","C_Easy2_noise015","C_Easy2_noise02", ...
            "C_Difficult1_noise005","C_Difficult1_noise01","C_Difficult1_noise015","C_Difficult1_noise02", ...
            "C_Difficult2_noise005","C_Difficult2_noise01","C_Difficult2_noise015","C_Difficult2_noise02"
            ];

vectorSize = 64;
startOffset = 0;
numClasses = 3;

for fileIdx = 1:length(baseName)
    %Construct an instance of the Dataset object
    dataFile = strcat(path,baseName(fileIdx),"_data.txt");
    classFile = strcat(path,baseName(fileIdx),"_spikeClass.txt");
    timeFile = strcat(path,baseName(fileIdx),"_spikeTimes.txt");
    
    % Load data from files
    f = fopen(dataFile,'r');
    data = fscanf(f,"%f");
    fclose(f);
    
    f = fopen(classFile,'r');
    classInd = fscanf(f,"%f");
    fclose(f);
    
    f = fopen(timeFile,'r');
    spike_times = fscanf(f,"%f");
    fclose(f);
    
    %Get Spikes (bypass detecting)
    numSpikes = length(spike_times);
    spikes = zeros([numSpikes,vectorSize]);
    spikeCounts = zeros(1,numClasses);
    spikeInd = zeros(numClasses,1500);
    
    for spikeIdx=1:numSpikes
        start = spike_times(spikeIdx)+startOffset;
        spikes(spikeIdx,:) = data(start:start+vectorSize-1);
    
        class = classInd(spikeIdx);
        spikeCounts(class) = spikeCounts(class)+1;
        spikeInd(class,spikeCounts(class)) = spikeIdx;
    end
    
    save(strcat(baseName(fileIdx),"_spikes.mat"),"spikes","classInd");
end