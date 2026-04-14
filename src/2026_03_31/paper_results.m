axisFontSize = 10; %originally 20
titleFontSize = 10; %originally 25
legendFontSize = 10; %originally 20

minComponents = 2;
maxComponents = 10;
numFiles = 20; 

feMethods = ["HT","WHT"];
%threshMethods = ["Linear (FP)", "Bin (FP)", "Bin (8-Bit)"];
trainSizes = ["N=30","N=60","N=90","N=120","N=150","N=300","N=600","N=900","N=1200","N=2100"];

%AVG = zeros(maxComponents-minComponents+1,length(trainSizes),length(feMethods));
%STD = zeros(length(threshMethods),length(trainSizes),length(feMethods));

directories = ["HT_slidingWindow/","WHT_slidingWindow/"];
%matFiles = ["FSDE_fsa_linear.mat","FSDE_fsa_bin.mat","FSDE_fsa_bin_quant.mat"; ...
%    "PCA_FD_fsa_linear.mat","PCA_FD_fsa_bin.mat","PCA_FD_fsa_bin_quant.mat"];

lowNoiseIndices = [1,3,5,7];
highNoiseIndices = [2,4,6,8];

%% Plot Parameters
hues = linspace(0,1,length(trainSizes)+1); hues(end) = [];  % remove redundant last hue = 1
colors = zeros(length(trainSizes),3);
for colorIdx=1:length(trainSizes)
    colors(colorIdx,:) = hsv2rgb([hues(colorIdx),1,0.8]); 
end

%% Generate Plots
for numComponents = minComponents:maxComponents
    classificationResults = figure('Name',"results_"+numComponents);     
    
    for fileIdx=1:numFiles
        subplot(5,4,fileIdx)
        colororder(colors);
        
        AVG = zeros(length(directories),length(trainSizes));
        STD = zeros(length(directories),length(trainSizes));
        for dirIdx=1:length(directories)
            fileName = strcat("FWHT"+numComponents,"_slidingWindow_all_kmeans.mat");
            load(strcat(directories(dirIdx),fileName));
            AVG(dirIdx,:) = avgFScoreUnseen(fileIdx,:);
            STD(dirIdx,:) = stdFScoreUnseen(fileIdx,:);
        end %directories

        hb = bar(AVG);
        hold on
        for trainIdx = 1:length(trainSizes)
            xpos = hb(trainIdx).XEndPoints;
            errorbar(xpos,AVG(:,trainIdx),STD(:,trainIdx)./3.3,'LineStyle','none', ...
                'Color','k','LineWidth',1);
        end %trainIdx
    
        set(gca,'xticklabel',feMethods);
        ylim([60,100]);
        ylabel("F-Score","FontSize",axisFontSize);
        %xlabel("Coeff Selection Method","FontSize",axisFontSize);
        title(baseName(fileIdx),"FontSize",titleFontSize);
        ax = gca(gcf);
        ax.XAxis.FontSize = axisFontSize;
        set(ax.YAxis,'FontSize',axisFontSize);
        
    end %fileIdx
    sgtitle("Sliding Consecutive Coefficients");  
    set(classificationResults,"Position",[0,0,940,625]);
    imwrite(getframe(classificationResults).cdata,strcat("ht_wht_",get(classificationResults,'Name'),".png"));
end %numComponents

%STD = STD.*100;
%AVG = AVG.*100;


% for i=1:length(feMethods)
%     subplot(1,2,i)
%     colororder(colors);
%     hb = bar(AVG(:,:,i));
%     hold on
%     for j = 1:length(trainSizes)
%         xpos = hb(j).XEndPoints;
%         errorbar(xpos,AVG(:,j,i),STD(:,j,i)./3.3,'LineStyle','none', ...
%             'Color','k','LineWidth',1);
%     end
% 
%     set(gca,'xticklabel',threshMethods);
%     ylim([80,100]);
%     ylabel("F-Score","FontSize",axisFontSize);
%     xlabel("Thresholding Method","FontSize",axisFontSize);
%     title(feMethods(i)+newline+newline,"FontSize",titleFontSize);
%     ax = gca(gcf);
%     ax.XAxis.FontSize = axisFontSize;
%     set(ax.YAxis,'FontSize',axisFontSize);
% 
% end

% legend(trainSizes,'Location','bestoutside','FontSize',legendFontSize,'NumColumns',4)
% %legend(trainSizes,'FontSize',legendFontSize,'Location','southoutside');
% fig = gcf;
% %fig.Position(3)+250;
% %sgtitle(" ");
