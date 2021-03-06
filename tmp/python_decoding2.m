cd ../Python/Brain_DL
fileinfos=dir("res*.mat")
file_num=size(fileinfos,1)
cd ../../tmp_results/
iter=10
acc=zeros(file_num,iter)

for i=1:file_num

    load(fileinfos(i).name)
    YY=(D'*X')';
    X=YY;
    for j=1:iter
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
        acc(i,j)=(sum(pred==label_test)/size(pred,1))*100
    end
end

save("python_decoding_results.mat","fileinfos","acc")
