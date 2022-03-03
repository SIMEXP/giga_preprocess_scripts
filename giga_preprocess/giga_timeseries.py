"""
NIAK timeseries extraction and confound removal per subject.

To do:
- preventad - MINC data to nifti.
"""
import argparse
import re
import tarfile
from pathlib import Path

import pandas as pd
import numpy as np
from nilearn.image import load_img
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.connectome import ConnectivityMeasure
from nilearn.masking import compute_brain_mask
import sklearn
from templateflow import api as tflow
from templateflow import conf as tf_conf

from datetime import datetime



TEMPLATEFLOW_HOME = "./data/raw/customed_templateflow"
GIGA_PATTERN = r"fmri_s(ub)?(?P<sub>[A-Za-z0-9]*)_(sess(?P<ses>[A-Za-z0-9]*))?(_task)?(?P<task>rest|memory|FU[0-9]*|BL[0-9]*)(_)?(run(?P<run>[0-9]{,3}))?"

NIAK_CONFOUNDS = ["motion_tx", "motion_ty", "motion_tz",
                  "motion_rx", "motion_ry", "motion_rz",
                  "wm_avg", "vent_avg", "slow_drift"]

ATLAS_METADATA = {
    'mist': {
        'source': "user_define",
        'templates' : ['MNI152NLin2009bSym'],
        'resolutions': [3],
        'atlas': 'MIST',
        'description_pattern': "{dimension}",
        'dimensions': [7, 12, 20, 36, 64, 122, 197, 325, 444, 'ROI'],
        'atlas_parameters': ['resolution', 'desc'],
        'label_parameters': ['resolution','desc'],
        },
    'schaefer': {
        'source': "templateflow",
        'templates' : ['MNI152NLin2009cAsym', 'MNI152NLin6Asym'],
        'resolutions': [1, 2],
        'atlas': 'Schaefer2018',
        'description_pattern': "{dimension}Parcels7Networks",
        'dimensions': [100, 200, 300, 400, 500, 600, 800, 1000],
        'atlas_parameters': ['resolution', 'desc'],
        'label_parameters': ['desc'],
        },
    'segmented_difumo': {
        'source': "user_define",
        'templates' : ['MNI152NLin2009cAsym'],
        'resolutions': [2, 3],
        'atlas': 'DiFuMo',
        'description_pattern': "{dimension}dimensionsSegmented",
        'dimensions': [64, 128, 256, 512, 1024],
        'atlas_parameters': ['resolution', 'desc'],
        'label_parameters': ['resolution','desc'],
        },
    }

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description="", epilog="""
    Timeseries extraction on a preprocessed file from giga processing datasets.
    """)
    parser.add_argument(
        "-i", "--input", required=True, help=""
        "NIAK giga processing dataset root.",
    )
    parser.add_argument(
        "-d", "--dataset", required=True, help=""
        "NIAK giga processing dataset name {adni, ccna, cimaq, oasis, preventad}",
    )
    parser.add_argument(
        "-s", "--scan", required=True, help=""
        "File name of a preprocessed scan from the dedicated NIAK giga process dataset {adni, ccna, cimaq, oasis, preventad}",
    )
    parser.add_argument(
        "-o", "--output", required=True, help=""
        "Output path root.",
    )
    parser.add_argument(
        "-a", "--atlas", required=False, default=-1, help="""
        Atlas for timeseries extraction. Valid resolutions available are {schaefer, mist, segmented_difumo}, -1 for all (default: -1).
        When specific atlas is requested, the output directory and tar file will be suffixed with the name of the atlas.""",
    )
    return parser


def fetch_atlas_path(atlas_name, template, resolution, description_keywords):
    """
    Generate a dictionary containing parameters for TemplateFlow quiery.

    Parameters
    ----------
    atlas_name : str
        Atlas name. Must be a key in ATLAS_METADATA.

    template : str
        TemplateFlow template name.

    resolution : int
        TemplateFlow template resolution.

    description_keywords : dict
        Keys and values to fill in description_pattern.
        For valid keys check relevant ATLAS_METADATA[atlas_name]['description_pattern'].

    Return
    ------
    sklearn.utils.Bunch
        Containing the following fields:

        maps : str
            Path to atlas map.

        labels : pandas.DataFrame
            The corresponding pandas dataframe of the atlas

        type : str
            'dseg' (for NiftiLabelsMasker) or 'probseg' (for NiftiMapsMasker)
    """

    cur_atlas_meta = ATLAS_METADATA[atlas_name].copy()

    if cur_atlas_meta['source'] != "templateflow":
        tf_conf.init_layout()

    img_parameters = generate_templateflow_parameters(cur_atlas_meta, "atlas", resolution, description_keywords)
    label_parameters = generate_templateflow_parameters(cur_atlas_meta, "label", resolution, description_keywords)
    img_path = tflow.get(template, raise_empty=True, **img_parameters)
    img_path = str(img_path)
    label_path = tflow.get(template, raise_empty=True, **label_parameters)

    labels = pd.read_csv(label_path, delimiter="\t")
    atlas_type = img_path.split('_')[-1].split('.nii.gz')[0]
    return sklearn.utils.Bunch(maps=img_path, labels=labels, type=atlas_type)


def generate_templateflow_parameters(cur_atlas_meta, file_type, resolution, description_keywords):
    """
    Generate a dictionary containing parameters for TemplateFlow quiery.

    Parameters
    ----------
    cur_atlas_meta : dict
        The current TemplateFlow competable atlas metadata.

    file_type : str {'atlas', 'label'}
        Generate parameters to quiry atlas or label.

    resolution : int
        Templateflow template resolution.

    description_keywords : dict
        Keys and values to fill in description_pattern.
        For valid keys check relevant ATLAS_METADATA[atlas_name]['description_pattern'].

    Return
    ------
    dict
        A dictionary containing parameters to pass to a templateflow query.
    """
    description = cur_atlas_meta['description_pattern']
    description = description.format(**description_keywords)

    parameters_ = {key: None for key in cur_atlas_meta[f'{file_type}_parameters']}
    parameters_.update({'atlas': cur_atlas_meta['atlas'], 'extension': ".nii.gz"})
    if file_type == 'label':
        parameters_['extension'] = '.tsv'
    if parameters_.get('resolution', False) is None:
        parameters_['resolution'] = resolution
    if parameters_.get('desc', False) is None:
        parameters_['desc'] = description
    return parameters_


def create_atlas_masker(atlas_name, description_keywords, template='MNI152NLin2009cAsym', resolution=2, nilearn_cache=""):
    """Create masker given metadata.

    Parameters
    ----------
    atlas_name : str
        Atlas name. Must be a key in ATLAS_METADATA.

    description_keywords : dict
        Keys and values to fill in description_pattern.
        For valid keys check relevant ATLAS_METADATA[atlas_name]['description_pattern'].

    template : str
        TemplateFlow template name.

    resolution : str
        TemplateFlow template resolution.
    """
    atlas = fetch_atlas_path(atlas_name,
                             resolution=resolution,
                             template=template,
                             description_keywords=description_keywords)

    atlas_desc = atlas.maps.split('desc-')[-1].split('_')[0]

    if atlas.type == 'dseg':
        masker = NiftiLabelsMasker(atlas.maps, detrend=True)
    elif atlas.type == 'probseg':
        masker = NiftiMapsMasker(atlas.maps, detrend=True)
    if nilearn_cache:
        masker = masker.set_params(memory=nilearn_cache, memory_level=1)
    labels = list(range(1, atlas.labels.shape[0] + 1))
    return masker, labels, atlas_desc


def niak2bids(niak_filename, dataset_name):
    """Parse niak file name to BIDS entities."""
    print("\t" + niak_filename)
    compile_name = re.compile(GIGA_PATTERN)
    return compile_name.search(niak_filename).groupdict()


def bidsish_timeseries_file_name(file_entitiles):
    """Create a BIDS-like file name to save extracted timeseries as tsv."""
    base = f"sub-{file_entitiles['sub']}_"

    if file_entitiles.get('ses', False):
        if file_entitiles.get('ses', False) is not None:
            base += f"ses-{file_entitiles['ses']}_"

    base += f"task-{file_entitiles['task']}_"

    if file_entitiles.get('run', False):
        if file_entitiles.get('run', False) is not None:
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
                                compression="gzip")
    else:
        print("{} does not exists, skipping!".format(confounds_path))
        return -1

    # make sure all fields exist
    if confounds.columns.isin(NIAK_CONFOUNDS).all():
        return -1
    else:
        confounds = confounds[NIAK_CONFOUNDS]

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
    if file_entitiles.get('ses', False):
        if file_entitiles.get('ses', False) is not None:
            session = f"ses-{file_entitiles['ses']}"
    else:
        session = None

    if session:
        timeseries_root_dir = output_dir / subject / session
    else:
        timeseries_root_dir = output_dir / subject
    timeseries_root_dir.mkdir(parents=True, exist_ok=True)
    return timeseries_root_dir


def log_time():
    now = datetime.now()
    return now.strftime("%H:%M:%S")


def main():
    nilearn_cache = ""
    args = get_parser().parse_args()

    dataset = args.dataset
    scanfile = args.scan
    dataset_dir = f"{dataset}_preprocess"

    output_root_dir = Path(args.output)
    input_dir = Path(args.input)

    tf_conf.TF_HOME = Path(TEMPLATEFLOW_HOME)
    tf_conf.update(local=True)

    log_time()
    print("#### {} ####".format(dataset_dir))
    preprocessed_data_dir = input_dir / dataset_dir / "resample"
    fmri_path = preprocessed_data_dir / scanfile
    if fmri_path.exists():
        print(fmri_path)
    else:
        print(f"File not found: {fmri_path}")
        return -1

    atlas_names = list(ATLAS_METADATA.keys()) if isinstance(args.atlas, int) else [args.atlas]
    print(f"process atlas: {atlas_names}")

    dataset_name = f"dataset-{dataset}"

    output_dir = output_root_dir / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    file_entitiles = niak2bids(fmri_path.name, dataset)
    confounds = load_niak_confounds(fmri_path)
    if isinstance(confounds, int):
        print("Confounds cannot be loaded correctly.")
        return confounds

    timeseries_output_dir = create_timeseries_root_dir(output_dir,
        file_entitiles)
    fmri_nii = load_img(str(fmri_path))
    mask_img = compute_brain_mask(fmri_nii, memory=nilearn_cache)
    output_filename_root = bidsish_timeseries_file_name(
                    file_entitiles)

    for atlas_name in atlas_names:
        print("-- {} --".format(atlas_name))

        if atlas_name == "mist":
            template = "MNI152NLin2009bSym"
            resolution = 3
        else:
            template = "MNI152NLin2009cAsym"
            resolution = 2

        for dimension in ATLAS_METADATA[atlas_name]['dimensions']:
            # check if 3 files exists
            description_keywords = {"dimension": dimension}
            masker, labels, atlas_desc = create_atlas_masker(
                    atlas_name, description_keywords, template=template, resolution=resolution)

            processed = list(timeseries_output_dir.glob(f"{output_filename_root}atlas-{atlas_name}{atlas_desc}_desc-*.tsv"))
            if len(processed) == 3:
                print(f'{dimension}\tfiles exist; skipped.')
            else:
                now = log_time()
                print(f'{dimension}\t{now}')
                masker.set_params(mask_img=mask_img)
                raw_tseries = masker.fit_transform(
                    str(fmri_path))
                clean_tseries = masker.fit_transform(
                    str(fmri_path), confounds=confounds)
                for desc, tseries in zip(['raw', 'deconfound'], [raw_tseries, clean_tseries]):
                    output_filename = output_filename_root + \
                        f"atlas-{atlas_name}{atlas_desc}_desc-{desc}_timeseries.tsv"
                    timeseries = pd.DataFrame(tseries, columns=labels)
                    timeseries.to_csv(timeseries_output_dir / output_filename, sep='\t', index=False)
                corr_measure = ConnectivityMeasure(kind="correlation")
                connectome = corr_measure.fit_transform([clean_tseries])[0]
                connectome = pd.DataFrame(connectome, columns=labels, index=labels)
                connectome.to_csv(
                    timeseries_output_dir / output_filename.replace("timeseries", "connectome"),
                    sep='\t')
    now = log_time()
    print(f'finished\t{now}')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())

# run tests:
# pytest extract_timeseries.py
def test_niak2bids():
    """Check niak name parser."""
    case_1 = "fmri_sub130S5006_sess20121114_taskrestrun01_n.nii.gz"
    case_2 = "fmri_sub130S5006_sess20121114_taskrest_n.nii.gz"
    case_cimaq = "fmri_sub164965_rest_run1_n.nii.gz"
    assert niak2bids(case_1, 'adin').get("run", False) == "01"
    assert niak2bids(case_2, 'adin').get("run", False) is None
    assert niak2bids(case_cimaq, 'cimaq').get("run", False) == "1"
    assert niak2bids(case_cimaq, 'cimaq').get("ses", False) is False
