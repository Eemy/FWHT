function goodClusters = computeClusterConfidence(counts,thresh,rho,distances)
%Compute confidence in cluster centers based on the number of
%classifications

%counts: number of training points classified to cluster center
%distances: pairwise distances between each cluster center
%rho: rho values for each cluster center

    numClusters = length(counts);
    mostSimilar = zeros(1,numClusters);
    [~,ordrho] = sort(rho,"descend");
    % max_d=max(max(distances,[],"all"));
    
    mostSimilar(ordrho(1)) = ordrho(1);
    for ii=2:numClusters
        min_d = distances(ordrho(ii),ordrho(1));
        mostSimilar(ordrho(ii)) = ordrho(1);
        for jj=1:ii-1
            if(distances(ordrho(ii),ordrho(jj))< min_d)
                min_d = distances(ordrho(ii),ordrho(jj));
                mostSimilar(ordrho(ii))=ordrho(jj);
                
                %if(rho(mostSimilar(ordrho(jj))) > rho(ordrho(jj)) )
                %    mostSimilar(ordrho(ii)) = mostSimilar(ordrho(jj));
                %end

            end
        end
    end

    %Compare counts of centers sharing neighbors
    S = zeros(1,numClusters);
    for i=1:numClusters
        if mostSimilar(i) ~= 0
            S(i) = counts(i)/counts(mostSimilar(i));
        end
    end
    %disp(S);

    %% Prune Clusters
    goodClusters = ones(1,numClusters);
    for i=1:numClusters    
        if S(i) < thresh
            goodClusters(i) = 0;
        end
    end
end