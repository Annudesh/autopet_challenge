import os
import typing
from .logger import logger
import numpy as np

from nnunetv2.experiment_planning.plan_and_preprocess_api import (
    extract_fingerprints,
    plan_experiments,
    preprocess
)
from nnunetv2.experiment_planning.verify_dataset_integrity import verify_dataset_integrity
from nnunetv2.run.run_training import run_training
from nnunetv2.evaluation.find_best_configuration import find_best_configuration
from nnunetv2.ensembling.ensemble import ensemble_folders
from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data


from .constants import *

class Runner:
    """Class to handle nnunet training
    """
    def __init__(
        self,
        dataset_id: int = DATASET_ID,
        num_folds: int = 5,
        export_val_probs: bool=True,
        configs: list[str] = configs,
        maindir: str = MAINDIR,
    )->None:
        """Constructor

        Args:
            dataset_id (int, optional): id of single dataset for training. Defaults to DATASET_ID.
            num_folds (int, optional): number of folds for training. Defaults to 5.
            export_val_probs (bool, optional): export npz files to determine best config. Defaults to True.
            configs (list[str], optional): list of configs that are compared for training. Defaults to configs.
            maindir (str, optional): path to directory where all files are saved. Defaults to MAINDIR.
        """
        self.dataset_id = dataset_id
        self.num_folds = num_folds
        self.export_val_probs = export_val_probs
        self.configs = configs
        self.maindir = maindir
        
        logger.info('Initialized Runner...')
        
    
    def fingerprints_plan_preprocess(self) -> None:
        #TODO: could add more options here for more advanced plan+preprocess
        """Perform the fingerprint extraction, plan the experiments (for 4 configs) and preprocess the data
        """
        logger.info('Verifying dataset integrity...')
        verify_dataset_integrity(os.path.join(self.maindir,
                                              RAW,
                                              DATASET_FULL_NAME))
        dataset_id_lst = [self.dataset_id]
        logger.info('Fingerprint extraction...')
        extract_fingerprints(dataset_id_lst)
        
        logger.info('Planning experiments...')
        plan_experiments(dataset_id_lst)
        
        logger.info('Preprocessing the data')
        preprocess(dataset_ids=dataset_id_lst,
                   configurations=self.configs,
                   #TODO: investigate this
                   num_processes=[8,4,8,8]
                   )
        
    def train(self)->None:
        #TODO: could add more options here for advanced training
        """train all configs over all folds
        """
        for config in configs:
            for fold in np.arange(self.num_folds):
                logger.info('Training configuration %s at fold %i' % (config, fold))
                run_training(
                    dataset_name_or_id=str(self.dataset_id),
                    configuration=config,
                    fold=fold,
                    export_validation_probabilities=self.export_val_probs
                )
                         
    def find_best_config(self)->None:
        #TODO: figure out how (after training) to find the best configuration
        """After training, determines the best configuration for inference
        """
        logger.info('Determining the best configuration')
        find_best_configuration(dataset_name_or_id=str(self.dataset_id))
        #TODO: need to figure out how to extract the best configuration from the inferences.txt file
        
    def predict(self):
        #TODO: figure out how to predict using our best config
        logger.info('Predicting on test images...')
        return ...
        
    def ensemble(self):
        #TODO: find how to make an ensemble
        return ...
    
    def postprocess(self):
        #TODO: find how to use find_best_config to determine the post-processing
        return ...
    
    def run(self):
        #TODO: will go through eveyrthing in order
        self.fingerprints_plan_preprocess()
        self.train()
        self.find_best_config()
    
    