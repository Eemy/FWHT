function [clAssign,labelCounts] = distance_assign(distanceMethod,labels,centers,fSpace)
%Labels correspond to order centers come in. Label each spike according to
%the closest center. 

    % Classify data by nearest distance clustering
    numSpikes = size(fSpace,1);
    numCenters = size(centers,1);

    clAssign = zeros(1,numSpikes);
    labelCounts = zeros(1,numCenters);
    for i=1:numSpikes

        minDist = 0;
        minIdx = 0;
        for j=1:numCenters
            dist = 0;

            %Compute distance to center
            switch distanceMethod
                case "Manhattan"
                    dist = ComputeDistance.Manhattan(centers(j,:),fSpace(i,:));
                case "Euclidean"
                    dist = ComputeDistance.Euclidean(centers(j,:),fSpace(i,:));
            end

            %Classify with closest cluster center
            if j==1
                minDist = dist;
                minIdx = j;
            end
            if dist < minDist
                minDist = dist;
                minIdx = j;
            end
        end
        
        %Count the number of spikes classified to each found cluster
        labelCounts(minIdx) = labelCounts(minIdx) + 1;
        clAssign(i) = labels(minIdx);
    end

end