MAINDIR = 'maindir'
PRED = 'predictions'
RAW = 'raw'
RAW_ENV = 'nnUNet_raw'
PREPROCESSED = 'preprocessed'
PREPROCESSED_ENV = 'nnUNet_preprocessed'
RESULTS = 'results'
RESULTS_ENV = 'nnUNet_results'
DATASET_ID = 725
DATASET_NAME = 'AUTO'
IMGTR = 'imagesTr'
IMGTS = 'imagesTs'
LABTR = 'labelsTr'
PET_IMG_SUV = "PET_IMG_SUV"
CT_IMG = "CT_IMG"
CT_IMG_RES = "CT_IMG_RES"
SEG = "SEG"
PET_IMG = "PET_IMG"
DATASETTFI = "dataset.json"
DATASET_FULL_NAME = 'Dataset'+str(DATASET_ID)+'_'+DATASET_NAME
FIEXT = '.nii.gz'
PET = '0000'
CT = '0001'
OMP_NUM_THREADS = 'OMP_NUM_THREADS'
NUM_PROC = 'nnUNet_n_proc_DA'

INFERENCE_INFO_JSON = "inference_information.json"

class ENVKEYS:
    raw = RAW_ENV
    preprocessed = PREPROCESSED_ENV
    results = RESULTS_ENV
    num_threads = OMP_NUM_THREADS
    num_procs = NUM_PROC 
    

class MODES:
    M_2D = "2d"
    M_3D_FULLRES = "3d_fullres"
    M_3D_LOWRES = "3d_lowres"
    M_3D_CASCADE_FULLRES = "3d_cascade_fullres"

configs = [
    MODES.M_2D,
    MODES.M_3D_FULLRES,
    MODES.M_3D_LOWRES,
    MODES.M_3D_CASCADE_FULLRES
]

labeldict = {
    # PET Image in SUV
    PET_IMG_SUV : "SUV.nii.gz",
    
    # CT Image resampled to PET
    CT_IMG_RES : "CTres.nii.gz",
    
    # Original CT image
    CT_IMG : "CT.nii.gz",
    
    # Manual annotations of tumor lesions
    SEG : "SEG.nii.gz",
    
    # Original PET image as activity counts
    PET_IMG : "PET.nii.gz"
}