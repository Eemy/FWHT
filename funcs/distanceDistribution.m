function [distAvg,distAvgK] = distanceDistribution(dMat,spike_class,idx_k,trainingIndices)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    numSpikes = size(dMat,1);
    
    distances1 = [];
    distances2 = [];
    distances3 = [];
    distancesK1 = [];
    distancesK2 = [];
    distancesK3 = [];

    distAvg = zeros(3);
    distAvgK = zeros(3);
    counts = zeros(3);
    countsK = zeros(3);

    count=1;
    for i=1:numSpikes
        for j=i+1:numSpikes
            ii = trainingIndices(i);
            jj = trainingIndices(j);

            row = spike_class(ii);
            col = spike_class(jj);
            distAvg(row,col) = distAvg(row,col)+dMat(i,j);
            counts(row,col) = counts(row,col)+1;

            % if spike_class(ii) == spike_class(jj)
            %    class = spike_class(ii);
            %    distAvg(class) = distAvg(class)+dMat(i,j);
            %    counts(class) = counts(class)+1;
            % end

            if spike_class(ii) == spike_class(jj)
                class = spike_class(ii);
                switch class
                    case 1
                        distances1(end+1) = dMat(i,j);
                    case 2
                        distances2(end+1) = dMat(i,j);
                    case 3
                        distances3(end+1) = dMat(i,j);
                end            
            end

            if ismember(count,idx_k)
                row = spike_class(ii);
                col = spike_class(jj);
                distAvgK(row,col) = distAvgK(row,col)+dMat(i,j);
                countsK(row,col) = countsK(row,col)+1;

                if spike_class(ii) == spike_class(jj)
                    class = spike_class(ii);
                    switch class
                        case 1
                            distancesK1(end+1) = dMat(i,j);
                        case 2
                            distancesK2(end+1) = dMat(i,j);
                        case 3
                            distancesK3(end+1) = dMat(i,j);
                    end
                end
            end
            
            count = count+1;
        end
    end

    % figure;
    % subplot(1,3,1)
    % histogram(distances1(:));
    % title("Group 1");
    % subplot(1,3,2)
    % histogram(distances2(:));
    % title("Group 2");
    % subplot(1,3,3)
    % histogram(distances3(:));
    % title("Group 3");
    % 
    % figure;
    % subplot(1,3,1)
    % histogram(distancesK1(:));
    % title("Group 1");
    % subplot(1,3,2)
    % histogram(distancesK2(:));
    % title("Group 2");
    % subplot(1,3,3)
    % histogram(distancesK3(:));
    % title("Group 3");    

    disp(counts);
    disp(countsK);

    distAvg = distAvg./counts;
    distAvgK = distAvgK./countsK;
end