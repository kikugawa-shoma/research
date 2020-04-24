function updated_D=K_SVD_par(X,D,y)

D_tmp=D;

%parfor i=1:size(D,2)
for i = 1:size(D,2)
    disp(i)
    Q=(X(i,:)~=0);
    tmp_X=X;
    tmp_X(i,:)=[];
    tmp_D=D;
    tmp_D(:,i)=[];
    
    E=y-tmp_D*tmp_X;
%     if sum(Q) ~= 0
%         E_R=E(:,find(Q));
%         [U,~,~]=svds(E_R,1);
%         %svd means E_R=U*S*V';
%         D_tmp(:,i)=U(:)/norm(U(:));
%     end
    if sum(Q) ~= 0
        E_R=E(:,find(Q));
        [U,S,V]=svds(E_R,1);
        %svd means E_R=U*S*V';
        D_tmp(:,i)=U(:)/norm(U(:));
        X(i,find(Q)) = S*V;
    end

end

updated_D=D_tmp;

end
