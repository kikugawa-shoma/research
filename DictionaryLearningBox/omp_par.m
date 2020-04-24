function res_x=omp(y,D)
%%
%initializetion
r=y;
epsilon=zeros(size(D,2),1);
max_k=size(D,2);
S=NaN(max_k,1);
S_all=1:max_k;
S_all=S_all';
S_cut=[];
lambda=0.001;      %正則化係数 
sparce_degree=4;   %スパース度
%%
%main iteration
for k=1:sparce_degree
    %norm_r(k,1)=norm(r);
    %sweep
    for i=1:size(D,2)
        if sum(S==i)>0
            epsilon(i,1)=NaN;
        else
            epsilon(i,1)=norm(r)^2-(dot(D(:,i),r))^2/norm(D(:,i))^2;
        end
    end
    %update support
    
    [~,min_I]=min(epsilon(:,1));
    
    S(k,1)=min_I;
    S_cut=rmmissing(S);
    S_cut=sort(S_cut);
    
    %uodate provisional solution
    x_l=(D(:,S_cut)'*D(:,S_cut)+lambda*eye(size(D(:,S_cut),2)))\(D(:,S_cut)'*y);
    
    %update residual
    r=y-D(:,S_cut)*x_l;
    
end


res_x=zeros(size(D,2),1);
res_x(S_cut)=x_l;

end