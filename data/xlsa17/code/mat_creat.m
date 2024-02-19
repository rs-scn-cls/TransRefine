T=readtable('UCM_img.csv', 'Delimiter', ','); % change {dataset}_img.csv for other datasets
image_files=T{:,1};
Y=readtable('UCM_label.csv', 'Delimiter', ',', 'Format', '%f'); % change {dataset}_labe.csv for other datasets
labels=Y{:,1};
save('res101.mat','image_files', 'labels')
