function logistic(package, main_item, comp_item)
% global variants
handles.data = 0;
handles.textdata = 0;

if nargin < 1
    package='Smarket';
    main_item = 'Direction';
    comp_item = {'Lag1', 'Lag2', 'Lag3', 'Lag4', 'Lag5', 'Volume', 'Today', 'Direction'};
end

[handles.data, handles.textdata] = read_data(package);
fprintf('%s\n', 'Coefficient');
most_relative = func_summary(handles, main_item, comp_item);
fprintf('\n The most relative item is "%s".\n', handles.textdata{most_relative});
end

function [data, textdata] = read_data(package)
tmp = importdata(['./data/', package, '.txt']);
data = tmp.data;
textdata = tmp.textdata;
end

% Hypothesis value, sigmoid function
function res = func_hypothesis(X, theta)
tmp = theta(1) + theta(2) * X+theta(3)*X.^2;
res = 1./(1 + exp(-tmp));
end


function [ cost, gradient ] = func_cost( theta, x, y )
m=size(x,1);
hypothesis = func_hypothesis(x, theta);

%cost function & gradient updating
cost = -sum(log(hypothesis + 0.01).* y + (1 - y).* log(1 - hypothesis + 0.01)) / m;
gradient(1) = sum(hypothesis-y) / m;
gradient(2) = sum((hypothesis-y).* x) / m;
gradient(3) = sum((hypothesis-y).*x.^2) / m;
end

% Gradient decent with opzimization
function [optTheta,functionVal,exitFlag] = gradient_descent(x, y)
 options = optimset('GradObj', 'on', 'MaxIter', 100, 'Display', 'off');
 theta = [0;0;0];
 [optTheta,functionVal,exitFlag] = fminunc(@(theta)func_cost(theta, x, y), theta, options);
end

% calculate the standart error with theta[]
function res = func_stdError(x, y, theta)
res = 0.0;
for i=1:size(x, 1)
    res = res + (func_hypothesis(x(i), theta) - y(i))^2;
end
res = res / size(x, 1);
end

function res = func_errorRate(theta, x, y)
res = 0.0;
for i=1:size(x, 1)
    if round(func_hypothesis(x(i), theta)) == y(i)
        res = res + 1;
    end
end
res = res / size(x, 1) * 100;
end

function most_corr = func_summary(handles, main_item, comp_items)
% Find the main column
main_col = 0;
for i=1:size(handles.textdata, 2)
    if strcmp(handles.textdata{i}, main_item)
        main_col = i;
        break; 
    end
end
if main_col == 0
    error('找不到主元素！');
end

% Print the header of the table
fprintf('%15s%15s%15s%15s%15s\n', '', 'Estimate', 'Std-Error', 'Pr(>|z|)', 'Precision');

% calculate the coefficients and find the most relative column
min_p = 1000;
for i=1:size(comp_items, 2)
    fprintf('%15s', comp_items{i});
    for j=1:size(handles.textdata, 2)
        if strcmp(handles.textdata{j}, comp_items{i})
            
            [optTheta, ~, ~] = gradient_descent(handles.data(:,j), handles.data(:, main_col));
            fprintf('%15f', optTheta(2));
            
            stderr = func_stdError(handles.data(:,j), handles.data(:, main_col), optTheta);
            fprintf('%15f', stderr);
            
            % How to calculate z-value???
            %fprintf('%15s', 'undefined');
            
            [Pr_r, Pr_p]= corrcoef(handles.data(:,j), handles.data(:,main_col));
            fprintf('%15f', Pr_p(1, 2));
            if Pr_p < min_p
                most_corr = j;
                min_p = Pr_p;
            end
            
            errrate = func_errorRate(optTheta, handles.data(:,j), handles.data(:,main_col));
            fprintf('%15.3f%%\n', errrate);
            break;
        end
    end
end
end