classdef Dataset < handle
    %UNTITLED2 Summary of this class goes here
    %   Detailed explanation goes here

    properties(Access = public)
        numSpikes
        spike_class %numSpikes x 1
        spikeVecs %numSpikes x vectorSize
        spikeCounts % number of spikes for each class (1x3)
        spikeIndices %list of spike number for each class (3x1500)
        
        %Filled when splitDataCounts is ran
        trainingIdx %[[class1idx][class2idx][class3idx]]
        classIdx
    end

    properties(Constant)
        numClasses = 3;
    end 

    methods
        function obj = Dataset(path,baseName,vectorSize,startOffset,noiseLevel)
            %Construct an instance of the Dataset object
            dataFile = strcat(path,baseName,"_data.txt");
            classFile = strcat(path,baseName,"_spikeClass.txt");
            timeFile = strcat(path,baseName,"_spikeTimes.txt");
            
            % Load data from files
            f = fopen(dataFile,'r');
            data = fscanf(f,"%f");
            fclose(f);
            
            f = fopen(classFile,'r');
            obj.spike_class = fscanf(f,"%f");
            fclose(f);
            
            f = fopen(timeFile,'r');
            spike_times = fscanf(f,"%f");
            fclose(f);

            %
            %Get Spikes (bypass detecting)
            %  
            obj.numSpikes = length(spike_times);
            obj.spikeVecs = zeros([obj.numSpikes,vectorSize]);
            obj.spikeCounts = zeros(1,obj.numClasses);
            obj.spikeIndices = zeros(obj.numClasses,1500);
            for spikeIdx=1:obj.numSpikes
                start = spike_times(spikeIdx)+startOffset;
                obj.spikeVecs(spikeIdx,:) = data(start:start+vectorSize-1);
        
                class = obj.spike_class(spikeIdx);
                obj.spikeCounts(class) = obj.spikeCounts(class)+1;
                obj.spikeIndices(class,obj.spikeCounts(class)) = spikeIdx;
        
                if noiseLevel ~= 0
                    % noise_std = noiseLevel*max(abs(spikeVecs(spikeIdx,:)));
                    % noisy = noise_std*randn(1,vectorSize);
                    % spikeVecs(spikeIdx,:) = spikeVecs(spikeIdx,:)+noisy;
                
                    noisy_sig = awgn(obj.spikeVecs(spikeIdx,:),20*log10(1/noiseLevel),'measured');
                    obj.spikeVecs(spikeIdx,:) = noisy_sig;
                end
        
            end
        end

        function [trainingLength,classificationLength] = splitDataCounts(obj,counts)
            %Finding true length of training/classification (all classes)
            obj.trainingIdx = [];
            obj.classIdx = [];
            
            %Determine output array lengths
            trainingLength = 0;
            classificationLength = 0;
            for classNum=1:obj.numClasses
                if counts(classNum) ~= 0
                    trainingLength = trainingLength + counts(classNum);
                    classificationLength = classificationLength + (obj.spikeCounts(classNum)-counts(classNum));
                end
            end

            obj.trainingIdx = zeros(1,trainingLength);
            obj.classIdx = zeros(1,classificationLength);
            trainingCount = 1;
            classificationCount = 1;
            for classNum=1:obj.numClasses
                if counts(classNum) ~= 0
                    %Get spikeVecs indices of class (remove trailing 0s)
                    spikesInClass = obj.spikeCounts(classNum);
                    trainIndices = obj.spikeIndices(classNum,1:spikesInClass);
                    
                    %Find end of section for that class
                    endingTrainIdx = trainingCount+counts(classNum)-1;
                    endingClassIdx= classificationCount+(spikesInClass-counts(classNum))-1;
                    
                    %Get random indices and append to section
                    trainingIdx = trainIndices(randperm(spikesInClass,counts(classNum)) );
                    classIdx = setxor(trainingIdx,trainIndices);
                    obj.trainingIdx(trainingCount:endingTrainIdx) = sort(trainingIdx);
                    obj.classIdx(classificationCount:endingClassIdx) = sort(classIdx);
                    
                    %Update start of next class section
                    trainingCount = endingTrainIdx+1;
                    classificationCount = endingClassIdx+1;
                end
            end
        end

        function [trainData] = getRandomSpikes(obj,N,classNumber)
            spikesInClass = obj.spikeCounts(classNumber);
            classIndices = obj.spikeIndices(classNumber,1:spikesInClass);

            %Get random indices
            trainIdx = classIndices(randperm(spikesInClass,N));
            
            trainData = zeros(N,size(obj.spikeVecs,2));
            for i=1:length(trainIdx)
                trainData(i,:) = obj.spikeVecs(trainIdx(i),:);
            end
        end

        function [spikeVecs_down] = downsample(obj,factor,offset)
            if offset >= factor
                disp("Invalid offset value");
                return;
            end

            vectorSize = size(obj.spikeVecs,2);
            vectorSize_down = floor(vectorSize/factor);

            spikeVecs_down = zeros(obj.numSpikes,vectorSize_down);
            for i=1:vectorSize_down
                vectorSize_down(:,i) = obj.spikeVecs(:,(i-1)*factor+1+offset);
            end
        end

    end %end methods
end %end classdef