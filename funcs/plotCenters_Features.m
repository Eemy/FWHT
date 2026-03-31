function plotCenters_Features(fig,graphTitle,nRows,nCols,graphIndex,numGroups,data,trainingFeatures,centerArr)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

    groupColors = ['b','y','g','r','k'];
    centerColor = 'm';
    labelFontSize = 10; %50 in paper figure
    axesFontSize = 10; %45 in paper figure
    titleFontSize = 10; %55 in paper figure
    centerSize = 10; %20 in paper figure

    set(0,'CurrentFigure',fig);
    numComponents = size(data,3);
    numCenters = length(centerArr);

    %% Plot Training Dataset
    subplot(nRows,nCols,graphIndex)
    hold on
    for idx1=1:numGroups
        if numComponents == 1
            plot(data(idx1,:),'o','MarkerFaceColor',groupColors(idx1),'MarkerEdgeColor',groupColors(idx1));
        else
            plot(data(idx1,:,1),data(idx1,:,2),'o','MarkerFaceColor',groupColors(idx1),'MarkerEdgeColor',groupColors(idx1));
        end
    end
    % Plot Centers over Training Data
    for idx1=1:numCenters
        if numComponents == 1
            plot(trainingFeatures(centerArr(idx1)),'o','MarkerFaceColor',centerColor,'MarkerSize',centerSize);
        else
            plot(trainingFeatures(centerArr(idx1),1),trainingFeatures(centerArr(idx1),2),'o','MarkerFaceColor',centerColor,'MarkerSize',centerSize);
        end
    end
    title(graphTitle);
    xlabel('PC1','FontSize',labelFontSize)
    ylabel('PC2','FontSize',labelFontSize)
    ax = gca(gcf);
    ax.XAxis.FontSize = axesFontSize;
    ax.YAxis.FontSize = axesFontSize;
    hold off


    % %% Plot Unseen Data for Classification 
    % subplot(nRows,nCols,graphIndex+1)
    % hold on
    % for idx1=1:numGroups
    %     if numComponents == 1
    %         plot(classificationVecs(idx1,:),'o','MarkerFaceColor',groupColors(idx1));
    %     else
    %         plot(classificationVecs(idx1,:,1),classificationVecs(idx1,:,2),'o','MarkerFaceColor',groupColors(idx1));
    %     end
    % end
    % % Plot Centers over Unseen Data
    % for idx1=1:numCenters
    %     if numComponents == 1
    %         plot(trainingFeatures(centerArr(idx1)),'o','MarkerFaceColor',centerColor,'MarkerSize',10);
    %     else
    %         plot(trainingFeatures(centerArr(idx1),1),trainingFeatures(centerArr(idx1),2),'o','MarkerFaceColor',centerColor,'MarkerSize',10);
    %     end
    % end
    % title(strcat("Unseen Data: ",graphTitle));
    % hold off
end