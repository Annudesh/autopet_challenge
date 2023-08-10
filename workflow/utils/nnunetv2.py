import os
import typing
from .logger import logger
import numpy as np
import json

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
        
    def _change_2d_batch_size(
        self,
        new_batch_size: int = 2
    ) -> None:
        """Modify the nnUNetPlans.json

        Args:
            new_batch_size (int, optional): Batch size for the 2d config. Defaults to 2.
        """
        f_path = os.path.join(
            self.maindir,
            PREPROCESSED,
            DATASET_FULL_NAME,
            'nnUNetPlans.json'
        )
        with open(f_path, 'r') as f:
            data = json.load(f)
            data["configurations"]["2d"]["batch_size"] = new_batch_size

        os.remove(f_path)
        with open(f_path, 'w') as f:
            json.dump(data, f, indent=4)
        
    
    def _fingerprints_plan_preprocess(self) -> None:
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
        
        logger.info('Changing 2d batch size...')
        self._change_2d_batch_size()
        
        logger.info('Preprocessing the data')
        preprocess(dataset_ids=dataset_id_lst,
                   configurations=self.configs,
                   #TODO: investigate this
                   num_processes=[8,4,8,8]
                   )
        
    def _train(self)->None:
        #TODO: could add more options here for advanced training
        """train all configs over all folds
        """
        for config in configs:
            for fold in np.arange(self.num_folds):
                logger.info('Training configuration %s at fold %i' % (config, fold))
                #continue_training = config=='2d' and fold==0
                run_training(
                    dataset_name_or_id=str(self.dataset_id),
                    configuration=config,
                    fold=fold,
                    export_validation_probabilities=self.export_val_probs,
                    #continue_training = continue_training
                )

    def _read_json_dict(
        self,
        file: str
    ) -> dict:
        """Read the provided json file and output the dictionary it contains

        Args:
            file (str): path to the json file

        Returns:
            dict: dictionary contained in the json file
        """
        if not file.endswith('.json'):
            logger.exception("Provided file is not json")
        f = open(file)
        return json.load(f)
        
                         
    def _find_best_config(self)->None:
        #TODO: figure out how (after training) to find the best configuration
        """After training, determines the best configuration for inference
        """
        logger.info('Determining the best configuration')
        find_best_configuration(dataset_name_or_id=str(self.dataset_id))
        
        try:
            inference_info_json = os.path.join(
                self.maindir, 
                RESULTS,
                DATASET_FULL_NAME,
                INFERENCE_INFO_JSON
            )
            self.best_config = self._read_json_dict(inference_info_json)
        except:
            logger.exception("Could not read inference information...")
        
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
        self._fingerprints_plan_preprocess()
        self._train()
        self._find_best_config()
    
    