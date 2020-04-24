%preprocessing_mat
%[U,S,V] = svds(D_all.data,3530);
svd_parameter = [10,100:200:3500];
acc_svd = size(svd_parameter,2);
for i=1:size(svd_parameter,2)
    N =svd_parameter(i);
    D = U(:,1:N)*S(1:N,1:N)*V(:,1:N)';
    L_A_svd=D_all.label(D_all.label==3|D_all.label==4);
    D_A_svd=D(D_all.label==3|D_all.label==4,:);
    ind_tr=sort(randsample(size(D_A_svd,1),round(size(D_A_svd,1)*0.8)));
    ind_test=setdiff(1:size(D_A_svd,1),ind_tr);
    D_tr_svd=D_A_svd(ind_tr,:);
    L_tr_svd=L_A_svd(ind_tr);
    D_test_svd=D_A_svd(ind_test,:);
    L_test_svd=L_A_svd(ind_test,:);
    
    model = fitcsvm(D_tr_svd, L_tr_svd);
    pred = predict(model,D_test_svd);
    acc_svd(1,i) = (sum(pred==L_test_svd)/size(pred, 1))*100;
end

