clear
addpath('/files')
path_data = ['/files' filesep]
path_out  =  '/files/cimaq_mem_rest_output2/'
%A second ouput to make the rest the first subject 


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

input = '/files/bids_output/'

%opt_bids = 'task-memory_run'
bids_data = niak_grab_bids2(input);
sub_names = fieldnames(bids_data);
maxdim = 0
good_run = '';

for k = 1:numel(sub_names)



	files_in.(sub_names{k}).anat = bids_data.(sub_names{k}).anat;	
	
	%%Get the run from the memory tasks that has the most amount of volumes. 
	sub_run = bids_data.(sub_names{k}).fmri.sess4;
	for i = 1 :length( fieldnames(sub_run))
		task =  fieldnames(sub_run){i};
            	sub_run_str = bids_data.(sub_names{k}).fmri.sess4.(task);	
		if ~(length(strfind(sub_run_str, "run")) == 0)
			hdr = niak_read_vol(sub_run_str);
			dim1 = hdr.info.dimensions(4);
			
	                if (dim1 > maxdim)
                   	     good_run  = sub_run_str;
        	             maxdim = dim1;


			end
			files_in.(sub_names{k}).fmri.memory.run1 = good_run;

		end
        end
	maxdim = 0;

	%This is the memory task
	if isfield(bids_data.(sub_names{k}).fmri.sess4,'taskmemory')  
                files_in.(sub_names{k}).fmri.memory.run1 = bids_data.(sub_names{k}).fmri.sess4.taskmemory; 
	end

	%resting state task
        files_in.(sub_names{k}).fmri.rest.run1 = bids_data.(sub_names{k}).fmri.sess4.taskrest


end

files_in = rmfield(files_in, 'sub700293')

return

disp(files_in)

opt.psom.max_queued = 60;
opt.flag_verbose = 0;



opt.folder_out = [path_out]
[pipeline,opt_pipe] = niak_pipeline_fmri_preprocess(files_in,opt);
