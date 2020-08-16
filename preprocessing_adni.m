addpath('/files')
path_data = ['/files' filesep]
path_out  =  '/files/adni_preprocess_output'

input = 'bids_output/'
%% General
opt.size_output = 'quality_control'; % The amount of outputs that are generated by the pipeline. 'all' will keep intermediate outputs, 'quality_control' will only keep the quality control outputs. 
opt.slice_timing.flag_skip        = 1;% Skip the slice timing (0: don't skip, 1 : skip). Note that only the slice timing corretion portion is skipped, not all other effects such as FLAG_CENTER or FLAG_NU_CORRECT

%% Motion estimation (niak_pipeline_motion)
%opt.motion.session_ref  = 'BL00';
%% resampling in stereotaxic space
opt.resample_vol.interpolation = 'trilinear'; % The resampling scheme. The fastest and most robust method is trilinear. 
opt.resample_vol.voxel_size    = [3 3 3];     % The voxel size to use in the stereotaxic space
opt.resample_vol.flag_skip     = 0;           % Skip resampling (data will stay in native functional space after slice timing/motion correction) (0: don't skip, 1 : skip)

%% Linear and non-linear fit of the anatomical image in the stereotaxic
% space (niak_brick_t1_preprocess)
opt.t1_preprocess.nu_correct.arg = '-distance 75'; % Parameter for non-uniformity correction. 200 is a suggested value for 1.5T images, 75 for 3T images. If you find that this stage did not work well, this parameter is usually critical to improve the results.

%% Temporal filtering (niak_brick_time_filter)
opt.time_filter.hp = 0.01; % Cut-off frequency for high-pass filtering, or removal of low frequencies (in Hz). A cut-off of -Inf will result in no high-pass filtering.
opt.time_filter.lp = Inf;  % Cut-off frequency for low-pass filtering, or removal of high frequencies (in Hz). A cut-off of Inf will result in no low-pass filtering.

%% Regression of confounds and scrubbing (niak_brick_regress_confounds)
opt.regress_confounds.flag_wm = true;            % Turn on/off the regression of the average white matter signal (true: apply / false : don't apply)
opt.regress_confounds.flag_vent = true;          % Turn on/off the regression of the average of the ventricles (true: apply / false : don't apply)
opt.regress_confounds.flag_motion_params = true; % Turn on/off the regression of the motion parameters (true: apply / false : don't apply)
opt.regress_confounds.flag_gsc = false;          % Turn on/off the regression of the PCA-based estimation of the global signal (true: apply / false : don't apply)
opt.regress_confounds.flag_scrubbing = true;     % Turn on/off the scrubbing of time frames with excessive motion (true: apply / false : don't apply)
opt.regress_confounds.thre_fd = 0.5;             % The threshold on frame displacement that is used to determine frames with excessive motion in the scrubbing procedure

%% Spatial smoothing (niak_brick_smooth_vol)
opt.smooth_vol.fwhm      = 6;  % Full-width at maximum (FWHM) of the Gaussian blurring kernel, in mm.
opt.smooth_vol.flag_skip = 0;  % Skip spatial smoothing (0: don't skip, 1 : skip)



opt.psom.max_queued = 55;
opt.flag_verbose = 0;


opt_cn.crop_neck = 0.25;


csv_file = csv2cell('failed_subjects_crop_neck_func.csv');
failed_subjects = csv_file(2:end);


files_in = niak_grab_bids2(input);


for j = 1: numel(failed_subjects)
	
       	% disp(failed_subjects{j});

       	 sub = strsplit(failed_subjects{j},'_')(1){1};
	 sess = strsplit(failed_subjects{j},'_')(2){1};

	 anat = files_in.(sub).anat;

	 opt.tune(j).subject = sub;
	 opt.tune(j).param.t1_preprocess.crop_neck = 0.25


end




%exclude these subjects/sessions 
files_in.sub012S4026.fmri =  rmfield(files_in.sub012S4026.fmri,'sess20111207')
files_in = rmfield(files_in,'sub070S6229')

files_in.sub006S4713.fmri =  rmfield(files_in.sub006S4713.fmri,'sess20120514')
files_in.sub006S4713.fmri =  rmfield(files_in.sub006S4713.fmri,'sess20120813')
files_in.sub007S4272.fmri = rmfield(files_in.sub007S4272.fmri,'sess20180116')

%
files_in = rmfield(files_in,'sub021S4254')
files_in = rmfield(files_in,'sub021S4744')


tune = j+1
opt.tune(tune).subject = 'sub027S4919'
opt.tune(tune).param.t1_preprocess.crop_neck = 0.4

tune +=1 
opt.tune(tune).subject = 'sub129S6621'
%opt.tune(tune).param.t1_preprocess.crop_neck = 0.4
opt.tune(tune).param.t1_preprocess.nu_correct.arg = '-distance 200';


files_in =  rmfield(files_in,'sub007S4272')

%minctracc throws the following error volume 1 with value above threshold (0.000000).
files_in =  rmfield(files_in,'sub129S6621')

files_in =  rmfield(files_in,'sub002S0685')


opt.folder_out = path_out;
[pipeline,opt_pipe] = niak_pipeline_fmri_preprocess(files_in,opt);
 
 
