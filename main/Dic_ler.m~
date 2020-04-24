p=gcp('nocreate');
if isempty(p)
    p=parpool;
end

addpath C:\Users\ktmks\Documents\research\DictionaryLearningBox
y=D_all.data';

%% initialize dictionary D
atom_N=1000
D_size=[size(y,1),atom_N];
D=normalize(rand(D_size),1);

%% main iteration
input_N=size(y',1);
X=zeros(D_size(1,2),input_N);
K=5;                           %繰り返し回数
R=zeros(K,1);                  %コストを入れる配列
cor = zeros(K,1);

for k=1:K
    tic
    %Sparse Coding State
    parfor l=1:input_N
        disp(l)
        X(:,l)=omp_par(y(:,l),D);
    end
    
    X = fillmissing(X,'constant',0);
    R(k,1)=norm(y-D*X);
    tmp = corrcoef(y,D*X);
    cor(k,1) = tmp(1,2);

    %K-SVD Dictionary update state
    D=K_SVD_par(X,D,y);
    toc
end

%% Dictionary Learning decoding
D_A_dic=X(:,D_all.label==3|D_all.label==4)';
L_A_dic=D_all.label(D_all.label==3|D_all.label==4);
D_A_raw=D_all.data(D_all.label==3|D_all.label==4,:);
L_A_raw=D_all.label(D_all.label==3|D_all.label==4);
acc_dic=zeros(1,10);
acc_raw=zeros(1,10);
save("C:\Users\ktmks\Documents\research\tmp_results\Dic_ler_result"+num2str(atom_N)+".mat","D_A_dic","L_A_dic","acc_dic","R","cor","D","X","y")

for i=1:10
    ind_tr=sort(randsample(size(D_A_raw,1),round(size(D_A_raw,1)*0.8)));
    ind_test=setdiff(1:size(D_A_raw,1),ind_tr);
    
    D_tr_dic=D_A_dic(ind_tr,:);
    L_tr_dic=L_A_dic(ind_tr);
    D_test_dic=D_A_dic(ind_test,:);
    L_test_dic=L_A_dic(ind_test,:);
    
    model = fitcsvm(D_tr_dic, L_tr_dic);
    pred = predict(model,D_test_dic);
    acc_dic(1,i) = (sum(pred==L_test_dic)/size(pred, 1)) * 100;
   
    %% Standard SVM decoding
    D_tr_raw=D_A_raw(ind_tr,:);
    L_tr_raw=L_A_raw(ind_tr);
    D_test_raw=D_A_raw(ind_test,:);
    L_test_raw=L_A_raw(ind_test,:);
    
    model = fitcsvm(D_tr_raw, L_tr_raw);
    pred = predict(model,D_test_raw);
    acc_raw(1,i) = (sum(pred==L_test_raw)/size(pred, 1))*100;
end

