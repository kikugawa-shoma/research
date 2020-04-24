

load("C:\Users\ktmks\Documents\research\Python\Brain_DL\res_AtomN-1000_SparseDegree-10_MaxIter-30.mat")

L=label(label==3|label==4);
data=X(label==3|label==4,:);

ind_tr=sort(randsample(size(L,1), round(size(L,1)*0.8)));
ind_test=setdiff(1:size(L,1), ind_tr);

data_tr=X(ind_tr,:);
label_tr=L(ind_tr);
data_test=X(ind_test,:);
label_test=L(ind_test);

model=fitcsvm(data_tr,label_tr);
pred=predict(model,data_test);
acc=(sum(pred==label_test)/size(pred,1))*100

