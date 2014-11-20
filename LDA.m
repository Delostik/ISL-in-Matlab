function LDA(package)
file = importdata('./data/Smarket.txt');
data = file.data;
header = file.textdata;

main_item = 'Direction';
comp_items = {'Lag1', 'Lag2'};
func_summary(data, header, main_item, comp_items);
end

function func_summary(data, header, main_item, comp_items)
main_col = find(strcmp(header, main_item));
if ~main_col
    error('找不到主元素！');
end

% prepare matrix included selected features
sampleMatrix = [];
for i=1:size(comp_items, 2)
    col = find(strcmp(header, comp_items{i}));
    if col
        sampleMatrix = [sampleMatrix, data(:, col)];
    end
end
category = data(:,size(data, 2));

% iteration
t=size(sampleMatrix, 2);

for times=1:t-1
    eigenVector = func_eigenVector(sampleMatrix, category);
    sampleMatrix = sampleMatrix * eigenVector;
end

disp(sampleMatrix);
end

function eigvector = func_eigenVector(X, gnd)
[nSmp,nFea] = size(X); 
classLabel = unique(gnd); 
nClass = length(classLabel); 

sampleMean = mean(X); 
MMM = zeros(nFea, nFea); 
for i = 1:nClass
	index = find(gnd == classLabel(i)); 
	classMean = mean(X(index, :)); 
	MMM = MMM + length(index)*(classMean')*classMean; 
end 
W = X'*X - MMM; 
B = MMM - nSmp*(sampleMean')*sampleMean; 
 
W = (W + W');
B = (B + B');
 
[eigvector, eigvalue] = eigs(B,W, nClass - 1); 

for i = 1:size(eigvector,2) 
    eigvector(:,i) = eigvector(:,i)./norm(eigvector(:,i)); 
end 
end
