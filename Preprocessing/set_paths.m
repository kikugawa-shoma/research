function paths = set_paths(paths)
% set_paths - function to load paths for data and libraries
%
% ----------------------------------------------------------------------------------------
% Created by members of
%     ATR Intl. Computational Neuroscience Labs, Dept. of Neuroinformatics


%% Library paths:
% Set path to BDTb:
if ~exist('paths','var') || ~isfield(paths,'to_lib') || isempty(paths.to_lib)
    dirname = uigetdir(pwd,'Select root-directory of ''BDTB''');
    if isequal(dirname,0)
        error('Can''t find BDTB directory');
    end
    paths.to_lib = dirname;
end


%% Add path (if needed):
str = which('addpath_bdtb');
if isempty(str),    addpath(paths.to_lib);      end
str = which('fixMkDir');
if isempty(str),    addpath_bdtb;               end


%% Data paths:
to_dat = getFieldDef(paths,'to_dat','');
if isempty(to_dat)
    dirname = uigetdir(pwd,'Select root-directory of ''DATA''');
    if isequal(dirname,0)
        error('Can''t find DATA directory');
    end
    paths.to_dat = dirname;
end
paths.to_dat       = fixMkDir(paths.to_dat);

paths.to_realigned = [paths.to_dat 'fmri_raw\realigned\'];
paths.to_realigned = fixMkDir(paths.to_realigned);

paths.to_unaligned = [paths.to_dat 'fmri_raw\unaligned\'];
paths.to_unaligned = fixMkDir(paths.to_unaligned);

paths.to_mat       = [paths.to_dat 'fmri_mat\'];
paths.to_mat       = fixMkDir(paths.to_mat);

paths.to_logs      = [paths.to_dat 'logs\'];
paths.to_logs      = fixMkDir(paths.to_logs);
