classdef ComputeDistance
    %UNTITLED7 Summary of this class goes here
    %   Detailed explanation goes here

    %properties(Constant)
    %    Property1
    %end
    %Spectator: Cam

    methods(Static)
        function distance = Manhattan(vec1,vec2)
            distance = sum(abs(vec1-vec2));
        end

        function distance = Euclidean(vec1,vec2)
            distance = sum((vec1-vec2).^2);
            %distance = sqrt(distance);
        end

        function [dMat,idx_k,pairs,dij,d_k] = computeAllPairs(mode,spikeFeatures,percentCutoff)
            %Returns matrix with all pairwise Manhattan distances and
            %indices of pairs wihin the distance cutoff

            %Get size parameters
            numTrainingSpikes = size(spikeFeatures,1);
            trainingTotalPairs = numTrainingSpikes*(numTrainingSpikes-1)/2;
            cutOffIdx = round(trainingTotalPairs*percentCutoff/100);

            %Allocate arrays
            dMat = zeros(numTrainingSpikes);
            %dij = zeros(1,cutOffIdx);
            dij = zeros(1,trainingTotalPairs);
            %idx_k = zeros(1,cutOffIdx);
            pairs = zeros(trainingTotalPairs,2);
            
            %Compute Distances
            counter = 1;
            %cutOffMax = 0.0;
            %cutOffMaxIdx = 0;
            for i=1:numTrainingSpikes
                for j=i+1:numTrainingSpikes

                    distance = 0;
                    switch mode
                        case "Manhattan"
                            distance = sum(abs(spikeFeatures(i,:)-spikeFeatures(j,:)));
                        case "Euclidean"
                            distance = sum((spikeFeatures(i,:)-spikeFeatures(j,:)).^2);
                    end

                    dMat(i,j) = distance;
                    dMat(j,i) = distance;
                    pairs(counter,1) = i;
                    pairs(counter,2) = j;
                    
                    dij(counter) = distance;

                    % % Fill max "heap" -- THIS CODE IS TOO SLOW SINCE IT
                    % DOESN'T IMPLEMENT AN ACTUAL BINARY HEAP
                    % if counter < cutOffIdx
                    %     dij(counter) = distance;
                    %     idx_k(counter) = counter;
                    % elseif counter == cutOffIdx
                    %     dij(counter) = distance;
                    %     idx_k(counter) = counter;
                    %     [cutOffMax,cutOffMaxIdx] = findExtrema(1,dij);
                    % % Once filled, replace maximum with lower values
                    % elseif distance < cutOffMax
                    %     dij(cutOffMaxIdx) = distance;
                    %     idx_k(cutOffMaxIdx) = counter;
                    %     [cutOffMax,cutOffMaxIdx] = findExtrema(1,dij);
                    % end

                    counter = counter+1;
                end %end j
            end %end i

            [d_k,idx_k] = mink(dij,cutOffIdx); %alternative to max "heap"
        end %end computeAllPairs

        %% Fixed-point version of methods
        function [dMat,idx_k,pairs] = computeAllPairs_fi(mode,spikeFeatures,percentCutoff,wordLength,fractionLength,F)
            %Returns matrix with all pairwise Manhattan distances and
            %indices of pairs within the distance cutoff

            %Get size parameters
            numTrainingSpikes = size(spikeFeatures,1);
            trainingTotalPairs = numTrainingSpikes*(numTrainingSpikes-1)/2;
            cutOffIdx = round(trainingTotalPairs*percentCutoff/100);

            %Allocate arrays
            dMat = fi(zeros(numTrainingSpikes),0,wordLength,fractionLength,F);
            pairs = zeros(trainingTotalPairs,2);

            %dij = fi(zeros(1,cutOffIdx),1,wordLength,fractionLength,F);
            dij = fi(zeros(1,trainingTotalPairs),0,wordLength,fractionLength,F);
            %idx_k = zeros(1,cutOffIdx);
            
            %Compute distances
            counter = 1;
            %cutOffMax = 0.0;
            %cutOffMaxIdx = 0;
            for i=1:numTrainingSpikes
                for j=i+1:numTrainingSpikes

                    distance = 0;
                    switch mode
                        case "Manhattan"
                            distance = ComputeDistance.Manhattan(spikeFeatures(i,:),spikeFeatures(j,:));
                        case "Euclidean"
                            distance = ComputeDistance.Euclidean(spikeFeatures(i,:),spikeFeatures(j,:));
                    end

                    dMat(i,j) = distance;
                    dMat(j,i) = distance;
                    pairs(counter,1) = i;
                     pairs(counter,2) = j;
                    
                    dij(counter) = distance;

                    % % Fill max "heap" -- THIS CODE IS SLOW FOR LARGE N SINCE IT
                    % % DOESN'T IMPLEMENT AN ACTUAL BINARY HEAP
                    % if counter < cutOffIdx
                    %     dij(counter) = distance;
                    %     idx_k(counter) = counter;
                    % elseif counter == cutOffIdx
                    %     dij(counter) = distance;
                    %     idx_k(counter) = counter;
                    %     [cutOffMax,cutOffMaxIdx] = findExtrema(1,dij);
                    % % Once filled, replace maximum with lower values
                    % elseif distance < cutOffMax
                    %     dij(cutOffMaxIdx) = distance;
                    %     idx_k(cutOffMaxIdx) = counter;
                    %     [cutOffMax,cutOffMaxIdx] = findExtrema(1,dij);
                    % end

                    counter = counter+1;
                end %end j
            end %end i
            
            [~,idx_k] = mink(double(dij),cutOffIdx); %alternative to max "heap"
        end %end computeAllPairs

    end %end methods
end %end classdef