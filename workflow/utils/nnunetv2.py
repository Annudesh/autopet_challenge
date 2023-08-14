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
        
    # def _change_2d_batch_size(
    #     self,
    #     new_batch_size: int = 2
    # ) -> None:
    #     """Modify the nnUNetPlans.json

    #     Args:
    #         new_batch_size (int, optional): Batch size for the 2d config. Defaults to 2.
    #     """
    #     f_path = os.path.join(
    #         self.maindir,
    #         PREPROCESSED,
    #         DATASET_FULL_NAME,
    #         'nnUNetPlans.json'
    #     )
    #     with open(f_path, 'r') as f:
    #         data = json.load(f)
    #         data["configurations"]["2d"]["batch_size"] = new_batch_size

    #     os.remove(f_path)
    #     with open(f_path, 'w') as f:
    #         json.dump(data, f, indent=4)
        
    
    def _fingerprints_plan_preprocess(self) -> None:
        #TODO: could add more options here for more advanced plan+preprocess
        """Perform the fingerprint extraction, plan the experiments (for 4 configs) and preprocess the data
        """
        # logger.info('Verifying dataset integrity...')
        # verify_dataset_integrity(os.path.join(self.maindir,
        #                                       RAW,
        #                                       DATASET_FULL_NAME))
        dataset_id_lst = [self.dataset_id]
        logger.info('Fingerprint extraction...')
        extract_fingerprints(dataset_id_lst,
                             check_dataset_integrity=True)
        
        logger.info('Planning experiments...')
        plan_experiments(dataset_id_lst)
        
        # logger.info('Changing 2d batch size...')
        # self._change_2d_batch_size()
        
        logger.info('Preprocessing the data')
        preprocess(dataset_ids=dataset_id_lst,
                   configurations=self.configs,
                   #TODO: investigate this
                   num_processes=[4,8,8]
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
        
        logger.info('Read the best config file')
        
    def _predict(
        self,
        list_of_lists_or_source_folder: typing.Union[list[list[str]], str],
        output_folder: str,
        model_training_output_dir: str,
    ) -> None:
        """Predict using the model on the provided input images

        Args:
            list_of_lists_or_source_folder (typing.Union[list[list[str]], str]): path to input images or list of list of images
            output_folder (str): where predicted segmentations are saved
            model_training_output_dir (str): the model directory to use (with folds as subdirectories)
        """
        #TODO: figure out how to predict using our best config
        logger.info('Predicting on test images...')
        from nnunetv2.inference.predict_from_raw_data import predict_from_raw_data
        
        predict_from_raw_data(
            list_of_lists_or_source_folder=list_of_lists_or_source_folder, 
            output_folder=output_folder,
            model_training_output_dir=model_training_output_dir
        )
        
    def _predict_ensemble_postprocess(
        self,
        folds = (0,1,2,3,4)
    ) -> None:
        
        from nnunetv2.ensembling.ensemble import ensemble_folders
        from nnunetv2.postprocessing.remove_connected_components import apply_postprocessing_to_folder
        from nnunetv2.utilities.file_path_utilities import get_output_folder
        
        input_test_dir: str = os.path.join(
            self.maindir,
            RAW,
            IMGTS
        )
        output_pred_dir = os.path.join(
            self.maindir,
            RESULTS,
            DATASET_FULL_NAME
        )

        run_ensemble = len(self.best_config["best_model_or_ensemble"]["selected_model_or_models"]) > 1

        used_folds = folds
        
        output_folders = []
        for config in self.best_config["best_model_or_ensemble"]["selected_model_or_models"]:
            output_dir = os.path.join(output_pred_dir, f"pred_{config['configuration']}")
            output_folders.append(output_dir)

            model_folder = get_output_folder(
                self.dataset_id,
                config["trainer"], 
                config["plans_identifier"], 
                config["configuration"]
            )
            
            self._predict(
                list_of_lists_or_source_folder=input_test_dir,
                output_folder=output_pred_dir,
                model_training_output_dir=model_folder,
                use_folds=used_folds,
                save_probabilities=run_ensemble
            )

        if run_ensemble:
            ensemble_folders(
                list_of_input_folders=output_folders,
                output_folder=os.path.join(
                    output_pred_dir,
                    "ensemble_predictions"
                ),
                save_merged_probabilities=False
            )
            folder_for_pp = os.path.join(
                output_pred_dir,
                "ensemble_predictions"
            )
        else:
            folder_for_pp = output_folders[0]

        # apply postprocessing
        from batchgenerators.utilities.file_and_folder_operations import load_pickle

        pp_fns, pp_fn_kwargs = load_pickle(self.best_config["best_model_or_ensemble"]["postprocessing_file"])
        apply_postprocessing_to_folder(
            input_folder=folder_for_pp,
            output_folder=os.path.join(
                output_pred_dir,
                "ensemble_predictions_postprocessed"
            ),
            pp_fns=pp_fns,
            pp_fn_kwargs=pp_fn_kwargs,
            plans_file_or_dict=self.best_config["best_model_or_ensemble"]["some_plans_file"],
        )
    
    def run(self):
        """Run through the whole nnunet process
        """
        self._fingerprints_plan_preprocess()
        self._train()
        self._find_best_config()
        self._predict_ensemble_postprocess()
    