path = "C:\Users\ktmks\Documents\research\tmp_results" 

atom_N_list = [10,100,1000,10,100,1000]
cor_mat=cell(6,1)
%old
for i=1:3
    cor_mat(i)={zeros(atom_N_list(i),atom_N_list(i))}
    datapath=[path+"\"+"old"+"\Dic_ler_result"+num2str(atom_N_list(i))+".mat"]
    load(datapath)
    for j =1:atom_N_list(i)
        for k=1:atom_N_list(i)
            tmp=corrcoef(D(:,j),D(:,k));
            cor_mat{i}(j,k)=tmp(1,2);
        end
    end
end

for i=4:6
    cor_mat(i)={zeros(atom_N_list(i),atom_N_list(i))}
    datapath=[path+"\"+"new"+"\Dic_ler_result"+num2str(atom_N_list(i))+".mat"]
    load(datapath)
    for j =1:atom_N_list(i)
        for k=1:atom_N_list(i)
            tmp=corrcoef(D(:,j),D(:,k));
            cor_mat{i}(j,k)=tmp(1,2);
        end
    end
end
