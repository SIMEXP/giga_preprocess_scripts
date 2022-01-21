"""
NIAK timeseries extraction and confound removal.

Save the output to a hdf5 file.

To do:
- dataset desceiptions
- confound variables - descriptions of them
- store an equivalent to participants.tsv
- tests for the confounds loader and the motion expansion
"""
import re
import tarfile
import os
from pathlib import Path

import pandas as pd
import numpy as np
import nilearn
import nilearn.input_data
import nilearn.datasets

NIAK_PATTERN = r"fmri_sub(?P<sub>[A-Za-z0-9]*)_sess(?P<ses>[A-Za-z0-9]*)_task(?P<task>[A-Za-z0-9]{,4})(run(?P<run>[0-9]{2}))?"
NIAK_CONFOUNDS = ["motion_tx", "motion_ty", "motion_tz",
                  "motion_rx", "motion_ry", "motion_rz",
                  "wm_avg", "vent_avg", "slow_drift"]
ATLAS_METADATA = {
    'schaefer7networks': {'type': "static", 'resolutions': [100, 400, 1000], 'fetcher': "nilearn.datasets.fetch_atlas_schaefer_2018(n_rois={resolution}, resolution_mm=2)"},
    'difumo': {'type': "dynamic", 'resolutions': [128, 512, 1024], 'label_idx': 1, 'fetcher': "nilearn.datasets.fetch_atlas_difumo(dimension={resolution}, resolution_mm=2)"},
}


def create_atlas_masker(atlas_name, nilearn_cache=""):
    """Create masker of all resolutions given metadata."""
    if atlas_name not in ATLAS_METADATA.keys():
        raise ValueError("{} not defined!".format(atlas_name))
    curr_atlas = ATLAS_METADATA[atlas_name]
    curr_atlas['name'] = atlas_name

    for resolution in curr_atlas['resolutions']:
        atlas = eval(curr_atlas['fetcher'].format(resolution=resolution))
        if curr_atlas['type'] == "static":
            masker = nilearn.input_data.NiftiLabelsMasker(
                atlas.maps, detrend=True)
        elif curr_atlas['type'] == "dynamic":
            masker = nilearn.input_data.NiftiMapsMasker(
                atlas.maps, detrend=True)
        if nilearn_cache:
            masker = masker.set_params(memory=nilearn_cache, memory_level=1)
        # fill atlas info
        curr_atlas[resolution] = {'masker': masker}
        if isinstance(atlas.labels[0], tuple) | isinstance(atlas.labels[0], list):
            if isinstance(atlas.labels[0][curr_atlas['label_idx']], bytes):
                curr_atlas[resolution]['labels'] = [
                    label[curr_atlas['label_idx']].decode() for label in atlas.labels]
            else:
                curr_atlas[resolution]['labels'] = [
                    label[curr_atlas['label_idx']] for label in atlas.labels]
        else:
            if isinstance(atlas.labels[0], bytes):
                curr_atlas[resolution]['labels'] = [
                    label.decode() for label in atlas.labels]
            else:
                curr_atlas[resolution]['labels'] = [
                    label for label in atlas.labels]

    return curr_atlas


def niak2bids(niak_filename):
    """Parse niak file name to BIDS entities."""
    print("\t" + niak_filename)
    compile_name = re.compile(NIAK_PATTERN)

    return compile_name.search(niak_filename).groupdict()


def bidsish_timeseries_file_name(file_entitiles, atlas_name, resolution):
    """Create a BIDS-like file name to save extracted timeseries as tsv."""
    base = f"sub-{file_entitiles['sub']}_ses-{file_entitiles['ses']}_task-{file_entitiles['task']}_"
    if "run" in file_entitiles:
        base += f"run-{file_entitiles['run']}_"
    base += f"atlas-{atlas_name}_nroi-{resolution}_timeseries.tsv"

    return base


def fetch_h5_group(f, subject, session):
    """Determine which level the file is in."""
    if subject not in f:
        if session:
            group = f.create_group(f"{subject}/{session}")
        else:
            group = f.create_group(f"{subject}")
    elif session:
        if session not in f[f"{subject}"]:
            group = f[f"{subject}"].create_group(f"{session}")
        else:
            group = f[f"{subject}/{session}"]
    else:
        group = f[f"{subject}"]

    return group


def temporal_derivatives(data):
    """Compute first order temporal derivative terms by backwards differences."""
    data_deriv = np.tile(np.nan, data.shape)
    data_deriv[1:] = np.diff(data, n=1, axis=0)

    return data_deriv


def load_niak_confounds(fmri_path):
    "Load full expansion of motion, basic tissue, slow drift."
    confounds_path = str(fmri_path).replace(".nii", "_confounds.tsv")
    if os.path.exists(confounds_path):
        confounds = pd.read_csv(confounds_path, sep="\t",
                                compression="gzip")[NIAK_CONFOUNDS]
    else:
        print("{} does not exists, skipping!".format(confounds_path))
        return -1
    # add temporal derivatives
    motion_deriv = []
    for m in NIAK_CONFOUNDS[:6]:
        label = f"{m}_derivative1"
        deriv_m = temporal_derivatives(confounds[m].values)
        deriv_m = pd.DataFrame(deriv_m, columns=[label])
        confounds = pd.concat([confounds, deriv_m], axis=1)
        motion_deriv.append(label)

    # add power term of original and temporal derivatives
    all_motions = NIAK_CONFOUNDS[:6] + motion_deriv
    for m in all_motions:
        power2 = confounds[m].values ** 2
        power2 = pd.DataFrame(power2, columns=[f"{m}_power2"])
        confounds = pd.concat([confounds, power2], axis=1)
    # Derivatives have NaN on the first row
    # Replace them by estimates at second time point,
    mask_nan = np.isnan(confounds.values[0, :])
    confounds.iloc[0, mask_nan] = confounds.iloc[1, mask_nan]

    return confounds


def create_timeseries_root_dir(file_entitiles):
    """Create root directory for the timeseries file."""
    subject = f"sub-{file_entitiles['sub']}"
    session = f"ses-{file_entitiles['ses']}" if file_entitiles.get(
        'ses', False) is not None else None
    if session:
        timeseries_root_dir = output_dir / subject / session
    else:
        timeseries_root_dir = output_dir / subject
    timeseries_root_dir.mkdir(parents=True, exist_ok=True)

    return timeseries_root_dir


if __name__ == '__main__':
    datasets = ["adni_preprocess", "aibl_preprocess", "ccna_preprocess",
                "cimaq_preprocess", "oasis_preprocess", "preventad_preprocess"]
    output_root_dir = Path(f"/data/cisl/preprocessed_data/giga_timeseries/")
    nilearn_cache = ""

    for dataset in datasets:
        print("#### {} ####".format(dataset))
        preprocessed_data_dir = Path(
            f"/data/cisl/giga_preprocessing/preprocessed_data/{dataset}/resample/")
        for atlas_name in ATLAS_METADATA.keys():
            print("-- {} --".format(atlas_name))
            dataset_name = f"dataset-{dataset.replace('_preprocess', '')}_atlas-{atlas_name}"
            output_dir = output_root_dir / dataset_name
            output_dir.mkdir(parents=True, exist_ok=True)
            fmri_data = preprocessed_data_dir.glob("*.nii.gz")
            fmri_data = list(fmri_data)
            atlas_masker = create_atlas_masker(
                atlas_name, nilearn_cache=nilearn_cache)
            for fmri_path in fmri_data:
                file_entitiles = niak2bids(fmri_path.name)
                confounds = load_niak_confounds(fmri_path)
                if isinstance(confounds, int):
                    continue
                timeseries_root_dir = create_timeseries_root_dir(
                    file_entitiles)
                for resolution in atlas_masker['resolutions']:
                    masker = atlas_masker[resolution]['masker']
                    output_filename = bidsish_timeseries_file_name(
                        file_entitiles, atlas_name, resolution)
                    timeseries = masker.fit_transform(
                        str(fmri_path), confounds=confounds)
                    labels = atlas_masker[resolution]['labels']
                    timeseries = pd.DataFrame(timeseries, columns=labels)
                    timeseries.to_csv(timeseries_root_dir /
                                      output_filename, sep='\t', index=False)
        # tar the dataset
        with tarfile.open(output_root_dir / f"{dataset_name}.tar.gz", "w:gz") as tar:
            tar.add(output_dir, arcname=output_dir.name)


# run tests:
# pytest extract_timeseries.py
def test_niak2bids():
    """Check niak name parser."""
    case_1 = "fmri_sub130S5006_sess20121114_taskrestrun01_n.nii.gz"
    case_2 = "fmri_sub130S5006_sess20121114_taskrest_n.nii.gz"
    assert niak2bids(case_1).get("run", False) == "01"
    assert niak2bids(case_2).get("run", False) is None
