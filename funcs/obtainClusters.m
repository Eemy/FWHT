function consensusClusters = obtainClusters(allFeatures,counts)
%Sorts nearest clusters together from each run, then averages them
%   Detailed explanation goes here
    
    %addpath("../modules");
    numComponents = size(allFeatures,3);
    [~,~,C] = mode(counts);
    likelyNumClusters = max(C{1,1});
    consensusClusters = zeros(1,likelyNumClusters,numComponents);

    for i=1:length(counts)
        if counts(i) == likelyNumClusters
            if i==1
                consensusClusters = allFeatures(i,1:likelyNumClusters,:);
            else
                
                for j=1:likelyNumClusters %for each cluster
                    minDistance = 100000;
                    closestCluster = 1;
                    for k = 1:likelyNumClusters %find closest cluster
                        dist = ComputeDistance.Manhattan(allFeatures(i,j,:),consensusClusters(1,k,:));
                        if(minDistance > dist)
                            minDistance = dist;
                            closestCluster = k;
                        end
                    end% for k
                    
                    %Update consensus/averaged cluster
                    for k=1:numComponents
                        consensusClusters(1,closestCluster,k) = (consensusClusters(1,closestCluster,k)*(i-1)+allFeatures(i,j,k))/i;
                    end

                end% for j

            end
        end% if likelyNumClusters
    end% for i
end