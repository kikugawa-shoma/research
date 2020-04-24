function [D, P] = prepro_mat(data)
% Function for running 'decode' process
% Copy, modify, and run this function for your data files
%
% Input:
%   data - struct of 'D', or mat-filename written by 'make_fmri_mat'
% Output:
%   res.model       - names of used model
%   res.pred        - predicted labels
%   res.label       - defined labels
%   res.dec_val     - decision values
%   res.weight      - weights (and bias)
%   res.freq_table  - frequency table
%   res.correct_per - percent correct
%   D.data          - 2D matrix of any data ([time(sample) x space(voxel/channel)] format)
%   D.label         - condition labels of each sample ([time x ltype] format)
%   D.label_type    - name of each labeling type ({1 x ltype] format)
%   D.label_def     - name of each condition ({ltype}{1 x condition} format)
%   D.design        - design matrix of experiment ([time x dtype] format)
%   D.design_type   - name of each design type ({1 x dtype] format)
%   D.roi           - voxel included ROIs ([rtype x space] format)
%   D.roi_name      - name of each ROI ({rtype x 1} format)
%   D.stat          - statistics of each voxel/channel ([stype x space] format)
%   D.stat_type     - name of each statistical type ({stype} format)
%   D.xyz           - x,y,z-coordinate values of each voxel/channel ([3(x,y,z) x space] for
%   P.script_name   - name of performed script/function name (this file)
%   P.date_time     - date and time this function was performed
%   P.procs1        - name of pre-validation preprocessing functions
%   P.procs2        - name of within-validation preprocessing functions
%   P.models        - name of within-validation classification/regression models
%   P.paths         - paths of data and library
%   P.<function>    - parameters for each function
%
% ----------------------------------------------------------------------------------------
% Created by members of
%     ATR Intl. Computational Neuroscience Labs, Dept. of Neuroinformatics


%% Set parameters:
P.script_name = mfilename;
P.date_time   = datestr(now,'yyyy-mm-dd HH:MM:SS');


%% Set paths:
%P.paths.to_lib = '/home/dcn/fuchigami/software/BDTB/BDTB-1.2.2_full/';    % path to root of BDTB
%P.paths.to_dat = '/home/dbi-data2/processing_common/20/';    % path to root of DATA (maybe directory named 'sbj_id')
P.paths.to_lib = 'C:\Users\ktmks\Documents\sources\BDTB\BDTB-1.2.2_full\';    % path to root of BDTB
P.paths.to_dat = 'C:\Users\ktmks\Documents\my_sources\20\';    % path to root of DATA (maybe directory named 'sbj_id

% if absent, select by GUI in the process


%% procs1 pars - pre-validation preprocessing:
P.procs1 = {
%    'selectLabelType';
    'shiftData';
    'fmri_selectRoi'; 'selectChanByTvals';
    'reduceOutliers'; 'detrend_bdtb';
%    'highPassFilter';
    'averageBlocks';
%    'averageLabels';
    'normByBaseline'
};
% Defines what will be done in for procs1
% Processings are sequentially run

% Parameters of procs1:
P.selectLabelType.target      = 1;
P.fmri_selectRoi.rois_use     = {};  %%% roi %%%
%P.selectChanByTvals.num_chans = 20000;      %%% (<-nVoxels)
%P.selectChanByTvals.tvals_min = 0;     %%%
P.reduceOutliers.std_thres    = 6;
P.reduceOutliers.num_its      = 2;
P.shiftData.shift             = 2;        %%% (<- -shiftTR)
P.normByBaseline.base_conds   = 1:5;


%% procs2 pars - within-validation preprocessing:
P.procs2 = {
%    'selectTopFvals';
    'zNorm_bdtb';
    'selectConds';
%    'balanceLabels'
};
% Defines what will be done in for procs2
% Processings are sequentially run

% Parameters of procs2:
P.zNorm_bdtb.app_dim   = 2;     % along space
P.zNorm_bdtb.smode     = 1;
P.selectConds.conds    = 3:4;   %%%  %%%
P.balanceLabels.method = 2;     % adjust to min

%% Model pars - within-validation classification/regression parameters:
P.models = {'libsvm_bdtb'};
% Defines what modeling will be performed:
% Exps: libsvm_bdtb, svm11lin_bdtb(32bit only),
%       slr_var_bdtb, smlr_bdtb(use Optimization Toolbox), slr_lap_bdtb(use Optimization Toolbox)

% Parameters of models:
%P.slr_var_bdtb.scale_mode = 'each';
%P.slr_var_bdtb.mean_mode  = 'each';
%P.slr_var_bdtb.norm_sep   = 1;

% Parameters of 'corssValidate'





%% ----------------------------------------------------------------------------
%% Run functions (avoid changing):
% Check args:
if ~exist('data','var') || isempty(data)
    [data, path] = uigetfile({'*.mat','MAT-files (*.mat)'},'Select mat-file saved ''D''');
    if isequal(data,0)
        error('''D''ata-struct or mat-filename saved ''D'' must be specified');
    end
    data = fullfile(path, data);
end

% Load data:
if ischar(data)
    data = load(data);
    data = data.D;
    D    = data;
else
    D    = data;
end

% Load paths:
P.paths = set_paths(P.paths);

% Add paths (if needed):
str = which('addpath_bdtb');
if isempty(str),    addpath(P.paths.to_lib);    end
str = which('procSwitch');
if isempty(str),    addpath_bdtb;               end

% Start writing log:
log_name = [P.paths.to_logs 'log_' P.script_name '_' P.date_time(1:10) '.txt'];
diary(log_name);
fprintf('\n============================================================');
fprintf('\nscript: %s \tdate: %s\n', P.script_name, P.date_time);

% Run procs1 - pre-validation preprocessing:
[D, P] = procSwitch(D,P,P.procs1);

%Validate - proc2 & model:
%[res, P] = crossValidate(D,P,P.procs2,P.models);

% Print results:
%if isfield(res,'test'),     printResults(res.test(:,end));
%else                        printResults(res(:,end));             end

% End log:
diary off;
