function dotSize = plotRho(trainingFeatures,pairs,idx_k,spike_class,indices,centers)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

    numComponents = size(trainingFeatures,2);
    numSpikes = max(pairs(:,2));
    dotSize = zeros(1,numSpikes);

    pairTracking = zeros(3);
    for i=1:length(idx_k)
        ii = pairs(idx_k(i),1);
        jj = pairs(idx_k(i),2);

        dotSize(ii) = dotSize(ii)+1;
        dotSize(jj) = dotSize(jj)+1;
        
        row = spike_class(indices(ii));
        col = spike_class(indices(jj));
        pairTracking(row,col) = pairTracking(row,col)+1;
    end
    %disp(pairTracking);
    
    hold on
    for i=1:numSpikes
        if dotSize(i) ~= 0
            if numComponents == 1
                plot(trainingFeatures(i,1),'o','MarkerFaceColor','r','MarkerEdgeColor','k',"MarkerSize",dotSize(i))
            else
                plot(trainingFeatures(i,1),trainingFeatures(i,2),'o','MarkerFaceColor','r','MarkerEdgeColor','k',"MarkerSize",dotSize(i))                
            end
        end
    end
    centerColors = ['b','g','y','m'];
    for i=1:length(centers)
        actualIdx = centers(i);
        if numComponents == 1
            plot(trainingFeatures(actualIdx,1),'o','MarkerFaceColor',centerColors(i),'MarkerEdgeColor','k',"MarkerSize",dotSize(actualIdx))
        else
            plot(trainingFeatures(actualIdx,1),trainingFeatures(actualIdx,2),'o','MarkerFaceColor',centerColors(i),'MarkerEdgeColor','k',"MarkerSize",dotSize(actualIdx))                
        end    
    end
    hold off
end