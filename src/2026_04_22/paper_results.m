axisFontSize = 10; %originally 20
titleFontSize = 10; %originally 25
legendFontSize = 10; %originally 20

minComponents = 2;
maxComponents = 10;
numComponents = minComponents:maxComponents;
numFiles = 20; 

feMethods = ["HT","WHT"];
%threshMethods = ["Linear (FP)", "Bin (FP)", "Bin (8-Bit)"];
FE_legend = ["FE=2","FE=3","FE=4","FE=5","FE=6","FE=7","FE=8","FE=9","FE=10"];

%AVG = zeros(maxComponents-minComponents+1,length(trainSizes),length(feMethods));
%STD = zeros(length(threshMethods),length(trainSizes),length(feMethods));

dir = "man_ht_random_train_06/";
suffix = "_slidingWindow_all_Manhattan_kmeans.mat";
%directories = ["HT_slidingWindow/","WHT_slidingWindow/"];
%matFiles = ["FSDE_fsa_linear.mat","FSDE_fsa_bin.mat","FSDE_fsa_bin_quant.mat"; ...
%    "PCA_FD_fsa_linear.mat","PCA_FD_fsa_bin.mat","PCA_FD_fsa_bin_quant.mat"];

%lowNoiseIndices = [1,3,5,7];
%highNoiseIndices = [2,4,6,8];

%% Consolidate Data
CHT_avg = zeros(numFiles,length(FE_legend));
CHT_std = zeros(numFiles,length(FE_legend));
WHT_avg = zeros(numFiles,length(FE_legend));
WHT_std = zeros(numFiles,length(FE_legend));
for i=1:length(feMethods)
    if i==1
        feString = "HT";
        feMat = CHT_avg;
        stdMat = CHT_std;
    else
        feString = "FWHT";
        feMat = WHT_avg;
        stdMat = WHT_std;
    end

    for j = 1:length(numComponents)
        fileName = strcat(feString+numComponents(j),suffix);
        load(strcat(dir,fileName));
        feMat(:,j) = avgFScoreUnseen;
        stdMat(:,j) = stdFScoreUnseen;
    end
    
    if i==1
        CHT_avg = feMat;
        CHT_std = stdMat;
    else
        WHT_avg = feMat;
        WHT_std = stdMat;
    end
end

%% Plot Parameters
hues = linspace(0,1,length(FE_legend)+1); hues(end) = [];  % remove redundant last hue = 1
colors = zeros(length(FE_legend),3);
for colorIdx=1:length(FE_legend)
    colors(colorIdx,:) = hsv2rgb([hues(colorIdx),1,0.8]); 
end

%% Generate Plots
classificationResults = figure('Name',"f1scores");     
for fileIdx = 1:numFiles
    subplot(5,4,fileIdx)
    colororder(colors);

    FScore_results = [CHT_avg(fileIdx,:) ; WHT_avg(fileIdx,:)];
    FScore_std = [CHT_std(fileIdx,:) ; WHT_std(fileIdx,:)];
    hb = bar(FScore_results);
    hold on
    for feIdx = 1:length(numComponents)
        xpos = hb(feIdx).XEndPoints;
        errorbar(xpos,FScore_results(:,feIdx),FScore_std(:,feIdx),'LineStyle','none', ...
            'Color','k','LineWidth',1);
    end

    set(gca,'xticklabel',feMethods);
    ylabel("F-Score","FontSize",10);
    ylim([50,100]);
    title(baseName(fileIdx));
end
sgtitle("Sliding Consecutive Coefficients");
set(classificationResults,"Position",[0,0,1880,1250]);
imwrite(getframe(classificationResults).cdata,strcat(dir,"ht_wht_",get(classificationResults,'Name'),".png"));
