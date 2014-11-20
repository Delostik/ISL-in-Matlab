function Smarket_corr()
Smarket = importdata('./data/Smarket.txt');

cor = corr(Smarket.data);
[n, n]=size(cor);
fprintf('%10s', '');
for i=1:n
    fprintf('%10s', Smarket.textdata{i});
end
fprintf('\n');

for i=1:n
    fprintf('%10s', Smarket.textdata{i});
    disp(cor(i,1:n));
end

