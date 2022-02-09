"""
NIAK timeseries extraction and confound removal.

Save the output to a hdf5 file.

To do:
- dataset desceiptions
- confound variables - descriptions of them
- store an equivalent to participants.tsv
- tests for the confounds loader and the motion expansion
"""
import argparse
import re
import tarfile
from pathlib import Path

import pandas as pd
import numpy as np
from nilearn.image import load_img
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.masking import compute_brain_mask
from templateflow import api as tflow
from templateflow import conf as tf_conf


DATASETS = ["adni_preprocess", "aibl_preprocess", "ccna_preprocess",
            "cimaq_preprocess", "oasis_preprocess", "preventad_preprocess"]

TEMPLATEFLOW_HOME = "./data/raw/templateflow"
INPUT_DIR = "/data/cisl/giga_preprocessing/preprocessed_data"
OUTPUT_ROOT = '/data/cisl/preprocessed_data/giga_timeseries/'
NIAK_PATTERN = r"fmri_sub(?P<sub>[A-Za-z0-9]*)_sess(?P<ses>[A-Za-z0-9]*)_task(?P<task>[A-Za-z0-9]{,4})(run(?P<run>[0-9]{2}))?"
NIAK_CONFOUNDS = ["motion_tx", "motion_ty", "motion_tz",
                  "motion_rx", "motion_ry", "motion_rz",
                  "wm_avg", "vent_avg", "slow_drift"]

ATLAS_METADATA = {
    'Schaefer2018': {
        'type': "dseg",
        'parcels': [100, 200, 300, 400, 500, 600, 800, 1000],
        'description': "{parcel}Parcels7Networks",
        'source': "templateflow"
        },
    'DiFuMo': {
        'type': "probseg",
        'parcels': [64, 128, 256, 512, 1024],
        'description': "{parcels}dimensionsSegmented",
        'source': "data/raw/segmented_difumo_atlases/"},
}


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="", epilog="""
    Timeseries extraction on giga processing datasets.
    """)

    parser.add_argument(
        "-d", "--dataset", required=True, help=""
        "NIAK giga processing dataset {adni, aibl, ccna, cimaq, oasis, preventad}",
    )

    parser.add_argument(
        "-a", "--atlas", required=False, default=-1, help=""
        "Atlas for timeseries extraction. Valid resolutions available are {Schaefer2018, DiFuMo}, -1 for all (default: -1)",
    )
    return parser


def create_atlas_masker(atlas_name, parcel, nilearn_cache=""):
    """Create masker of all number of parcel given metadata."""
    if atlas_name not in ATLAS_METADATA.keys():
        raise ValueError("{} not defined!".format(atlas_name))
    if parcel not in ATLAS_METADATA[atlas_name]['parcels']:
        raise ValueError("atlas-{} does not have map with {} number of parcels.".format(atlas_name, parcel))
    curr_atlas = ATLAS_METADATA[atlas_name]
    curr_atlas['atlas'] = atlas_name

    if curr_atlas['source'] == "templateflow":
        tf_conf.TF_HOME = Path(TEMPLATEFLOW_HOME)
    else:
        tf_conf.TF_HOME = Path(curr_atlas['source'])
    tf_conf.update()
    tf_conf.init_layout()

    desc = curr_atlas['description'].format(parcel=parcel)
    resolution = "02"
    atlas_path = tflow.get('MNI152NLin2009cAsym',
                            atlas=atlas_name,
                            resolution=resolution,
                            desc=desc)

    label_path = Path(str(atlas_path).replace("nii.gz", "tsv"))

    if curr_atlas['type'] == "dseg":
        masker = NiftiLabelsMasker(
            atlas_path, detrend=True)
    elif curr_atlas['type'] == "probseg":
        masker = NiftiMapsMasker(
            label_path, detrend=True)
    if nilearn_cache:
        masker = masker.set_params(memory=nilearn_cache, memory_level=1)

    return masker, label_path


def niak2bids(niak_filename):
    """Parse niak file name to BIDS entities."""
    print("\t" + niak_filename)
    compile_name = re.compile(NIAK_PATTERN)
    return compile_name.search(niak_filename).groupdict()


def bidsish_timeseries_file_name(file_entitiles):
    """Create a BIDS-like file name to save extracted timeseries as tsv."""
    base = f"sub-{file_entitiles['sub']}_ses-{file_entitiles['ses']}_task-{file_entitiles['task']}_"
    if  file_entitiles.get("run", False) is not None:
        base += f"run-{file_entitiles['run']}_"
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
    if Path(confounds_path).exists():
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


def create_timeseries_root_dir(output_dir, file_entitiles):
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

    output_root_dir = Path(OUTPUT_ROOT)
    input_dir = Path(INPUT_DIR)
    nilearn_cache = ""
    args = get_parser().parse_args()
    atlas_names = ATLAS_METADATA.keys() if args.atlas == -1 else [args.atlas]
    dataset = f"{args.dataset}_preprocess"

    print("#### {} ####".format(dataset))
    preprocessed_data_dir = input_dir / dataset / "resample"
    for atlas_name in atlas_names:
        print("-- {} --".format(atlas_name))
        dataset_name = f"dataset-{dataset.replace('_preprocess', '')}_atlas-{atlas_name}"
        output_dir = output_root_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)
        fmri_data = preprocessed_data_dir.glob("*.nii.gz")
        fmri_data = list(fmri_data)

        for fmri_path in fmri_data:
            file_entitiles = niak2bids(fmri_path.name)
            confounds = load_niak_confounds(fmri_path)
            if isinstance(confounds, int):
                continue
            timeseries_output_dir = create_timeseries_root_dir(output_dir,
                file_entitiles)
            fmri_nii = load_img(fmri_path)
            mask_img = compute_brain_mask(fmri_nii, memory=nilearn_cache)

            for parcel in ATLAS_METADATA[atlas_name]['parcels']:
                masker, label_path = create_atlas_masker(atlas_name, parcel, nilearn_cache)
                output_filename_root = bidsish_timeseries_file_name(
                    file_entitiles)
                atlas_desc = label_path.split('desc-')[-1].split('_')[0]

                masker.set_params(mask_img=mask_img)
                raw_tseries = masker.fit_transform(
                    str(fmri_path))
                clean_tseries = masker.fit_transform(
                    str(fmri_path), confounds=confounds)
                labels = pd.read_csv(label_path, sep='\t')
                for desc, tseries in zip(['raw', 'deconfound'], [raw_tseries, clean_tseries]):
                    output_filename = output_filename_root + \
                        f"atlas-{atlas_name}{atlas_desc}_desc-{desc}_timeseries.tsv"
                    timeseries = pd.DataFrame(tseries, columns=labels.iloc[:, 0])
                    timeseries.to_csv(timeseries_output_dir / output_filename, sep='\t', index=False)
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
