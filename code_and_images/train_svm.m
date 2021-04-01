%run('../vlfeat-0.9.20/toolbox/vl_setup')
load('pos_neg_feats_base.mat')

feats = cat(1,pos_feats,neg_feats);
labels = cat(1,ones(pos_nImages,1),-1*ones(neg_nImages,1));

lambda = 0.1;
[w,b] = vl_svmtrain(feats',labels',lambda);

fprintf('Classifier performance on train data:\n')
confidences = [pos_feats; neg_feats]*w + b;

[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, labels);

save('my_svm.mat','w','b')
%%
%run('../vlfeat-0.9.20/toolbox/vl_setup')
load('pos_neg_feats_valid.mat')

feats = cat(1,pos_feats,neg_feats);
labels = cat(1,ones(pos_nImages,1),-1*ones(neg_nImages,1));


fprintf('Classifier performance on validation data:\n')
confidences = [pos_feats; neg_feats]*w + b;

[tp_rate, fp_rate, tn_rate, fn_rate] =  report_accuracy(confidences, labels);

save('my_svm.mat','w','b')