addpath C:\Users\ktmks\Documents\research\Preprocessing\

data_dir='C:\Users\ktmks\Documents\my_sources\20\';

subj_idx = 1;
load('use_subj.mat')
decoding_list=list(1,:);
N=size(decoding_list,2);

%%
D_all.data=zeros(304*N,3530);
D_all.label=zeros(304*N,1);

for subj = decoding_list(1:N)
    subj_ID = ['subj',num2str(subj,'%0.3d')];
    for roi = 18
        mat_file_dir = [data_dir,num2str(subj,'%0.3d'),'/decoding/fmri_mat/my_ROI/roi_', num2str(roi),'/']; %/decoding_nonsm/fmri_mat/smooth_7

        [D, ~] = prepro_mat([mat_file_dir, char(subj_ID), '_fmri_roi_v7.mat'] );
        D_all.data( 304 * ( subj_idx - 1 ) + 1 : 304 * subj_idx , : )=D.data;
        D_all.label( 304 * ( subj_idx - 1 ) + 1 : 304 * subj_idx , : )=D.label;
        [~, num_channel] = size(D.data);
    end
    subj_idx = subj_idx + 1;
end

