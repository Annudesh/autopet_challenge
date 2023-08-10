import os
import typing
import random
import json
from rich.progress import track
import shutil
from .logger import logger

from .constants import *

class CompileData:
    """This class will create an object that will hold the paths
    to all of the datafiles provided by the autoPETDataset challenge.
    """
    def __init__(
        self,
        datadir : str,
        maindir : str
    ) -> None:
        """constructor

        Args:
            datadir (typing.Union[str, os.PathLike]): path to datadir (provided by autoPETChallenge)
            maindir (typing.Union[str, os.PathLike]): directory where raw, preprocessed and results will be saved
        """
        self.datadir = datadir
        self.maindir = maindir
        
        self.subdirs: list[str] = []
        for label in [RAW, PREPROCESSED, RESULTS]:
            try:
                subdir: str = os.path.join(self.maindir, label)
                os.makedirs(subdir)
                self.subdirs.append(subdir)
            except:
                logger.error(msg=f'Error creating directory {datadir}')
                raise Exception(f'Error creating directory {datadir}')
                
        self.data = self._getalldatafiles()
        
        
                
    def _listpatients(
        self
    ) -> list[str]:
        """provides a list of all the patients in the data directory

        Args: 
            supplied by autoPETChallenge

        Returns:
            tuple[str, ...]: list of all the patients
            
            
        """
        patients: list[str] = os.listdir(self.datadir)
        return patients

    def _liststudiesofpatient(
        self,
        patient:str
    ) -> list[str]:
        """list all the studies of the provided patient

        Args:
            patient (str): name of patient of interest

        Returns:
            tuple[str, ...]: list of all the studies
        """
        patientdir = os.path.join(self.datadir, patient)
        studies = os.listdir(patientdir)
        return studies
    
    def _makedatafiledictofstudy(
        self,
        datafiles: list[str]
    ) -> dict[str, str]:
        """make dictionary of the datafiles for the provided study

        Args:
            datafiles (tuple[typing.Union[str, os.PathLike], ...]): list of paths
            to all the datafiles provided by the specific study for the specific
            patient of interest

        Returns:
            dict[typing.Union[str, os.PathLike]]: dictionary allocating a label to
            each of the data files provided
        """
        def getlabel(finame: str) -> typing.Union[str, None]:
            """get the label of the provided file name

            Args:
                finame (str): Name of datafile

            Returns:
                str: label if it matches on of the 5 options
                None: did not match any
            """
            for label in labeldict:
                if finame == labeldict[label]:
                    return label
        
        datafiledict: dict[str, str] = {}
        for fi in datafiles:
            finame = os.path.split(fi)[-1]
            label = getlabel(finame)
            if not label:
                raise ValueError(f"{fi} does not match a file type we were expecting")
            datafiledict[label] = fi
            
        return datafiledict
        
    def _getdatafilesofstudy(
        self,
        patient:str,
        study:str
    ) -> list[str]:
        """Obtain a list of all the datafile paths for the study

        Args:
            patient (str): name of patient of interest
            study (str): name of study of interest

        Returns:
            tuple[str, ...]: list of all data file paths
        """
        
        studydir: str= os.path.join(self.datadir, patient, study)
        fis: list[str] = os.listdir(studydir)
        fis = list(map(lambda fi: os.path.join(studydir, fi), fis))
        return fis
        
    def _getalldatafiles(self) -> dict[str, dict[str, dict[str, str]]]:
        """Goes through the entire datadir and compiles a dictionary to describe it

        Returns:
            dict: dictionary containing the files of each study for each patient
            
        """
        files = {}
        for patient in self._listpatients():
            patientfiles = {}
            for study in self._liststudiesofpatient(patient):
                studyfiles = self._getdatafilesofstudy(patient, study)
                datafiledict = self._makedatafiledictofstudy(studyfiles)
                patientfiles[study] = datafiledict
            files[patient] = patientfiles
        
        return files
    
    def _getpatientdata(
        self,
        patient: str
    ) -> dict[str, dict[str, str]]:
        """Extract study data

        Args:
            patient (str): name of patient

        Returns:
            dict[typing.Union[str, os.PathLike]]: dictionary of patient data (all studies)
        """
        for _patient in self.data:
            if patient==_patient:
                return self.data[patient]
        raise KeyError(f"Patient {patient} could not be found")
    
    def _getpatientstudydata(
        self,
        patient: str,
        study: str
    ) -> dict[str, str]:
        """Extract study data for a specific patient

        Args:
            patient (str): name of patient
            study (str): name of patient

        Returns:    
            dict[typing.Union[str, os.PathLike]]: a dictionary of the relevant datafiles
        """
        patientdata = self._getpatientdata(patient)
        for _study in patientdata:
            if study == _study:
                return patientdata[study]
        raise KeyError(f"Study {study} could not be found for Patient {patient}")
        
    def _getnumberstudies(self) -> int:
        """Fetches the number of studies (across all patients) in the dataset

        Returns:
            int: Number of studies in the dataset
        """
        n=0
        for patient in self.data:
            patientdata = self.data[patient]
            n += len(list(patientdata.keys()))
        return n
            
    def _flattendata(self) -> list[list[str]]:
        """Make a list of all the studies (with their associated patient)

        Returns:
            tuple[typing.Union[str, os.PathLike], ...]: list of all studies
        """
        flatdata = []
        for patient in self.data:
            for study in self.data[patient]:
                flatdata.append([patient, study])
                
        return flatdata
    
    def _divtesttrain(
        self,
        testproportion: float = 0.10
    ) -> None:
        logger.info('Splitting testing and training data ...')
        """Splits the dataset into a training and testing set

        Args:
            testproportion (float, optional): Proportion of dataset that will be used for testing.
            Defaults to 0.10.
        """
        if not (testproportion>= 0) and (testproportion<= 1):
            raise ValueError('Test proportion must be between 0 and 1')
        
        switch_ix = int(self._getnumberstudies() *  testproportion)
        
        flatdata = self._flattendata()
        random.shuffle(flatdata)
        
        self.test = []
        self.train = []
        
        for ix, elem in enumerate(flatdata):
            if ix<= switch_ix:
                self.test.append(elem)
            else:
                self.train.append(elem)
                
    def _structurerawdir(self) -> None:
        """Make the raw training, test and label directories
        """
        self.datasetname = DATASET_FULL_NAME
        self.datasetdir = os.path.join(self.maindir,
                                       RAW,
                                       self.datasetname)
        if not os.path.isdir(self.datasetdir):
            os.makedirs(self.datasetdir)
        else:
            raise Exception(f'Could not create directory {self.datasetdir} since it already exists')
        
        for datadir in [IMGTR, IMGTS, LABTR]:
            os.makedirs(os.path.join(self.datasetdir,
                                    datadir))
    
    def _senddatatorawdir(self) -> None:
        """Copy the data to the appropriate directproes for MONAI nnunetv2
        """
        logger.info('Sending test data...')
        for elem in track(self.test, description='sending test data...'):
            patient, study = elem
            try:
                patientstudydata = self._getpatientstudydata(patient, study)
                pet, ct = patientstudydata[PET_IMG_SUV], patientstudydata[CT_IMG_RES]
                newpet = patient+"_"+study+"_"+PET+FIEXT
                newct = patient+"_"+study+"_"+CT+FIEXT
                for (img, newimg) in [[pet, newpet], [ct, newct]]:
                    shutil.copy(img,
                                os.path.join(self.datasetdir,
                                            IMGTS,
                                            newimg))
            except:
                logger.exception(f'Failed for patient: {patient} study: {study}')
        logger.info('Sending train data...')
        for elem in track(self.train, description='sending train data...'):
            self.numTrain = len(self.train)
            patient, study = elem
            try:
                patientstudydata = self._getpatientstudydata(patient, study)
                pet, ct, seg = patientstudydata[PET_IMG_SUV], patientstudydata[CT_IMG_RES], patientstudydata[SEG]
                newpet = patient+"_"+study+"_"+PET+FIEXT
                newct = patient+"_"+study+"_"+CT+FIEXT
                newseg = patient+"_"+study+FIEXT
                for (img, newimg, dest) in [[pet, newpet, IMGTR], [ct, newct, IMGTR], [seg, newseg, LABTR]]:
                    shutil.copy(img,
                                os.path.join(self.datasetdir,
                                            dest,
                                            newimg))
            except:
                self.numTrain -= 1
                logger.exception(f'Failed for patient: {patient} study: {study}')
                #raise Exception(f'Failed for patient: {patient} study: {study}')
        
    def _gendatasetjson(self) -> None:
        """generates the dataset.json file required my nnUNet
        """
        logger.info('Generating dataset.json file...')
        dataset_json = {
            "channel_names" : {
                "0" : "PET",
                "1" : "CT"
            },
            "labels": {
                "background" : 0,
                "tumour" : 1
            },
            "numTraining" : self.numTrain,
            "file_ending" : FIEXT            
        }
        
        dataset_path = os.path.join(self.datasetdir,
                                        DATASETTFI)
        with open(dataset_path , "w", encoding="utf-8") as f:
            json.dump(dataset_json, f, ensure_ascii=False, indent=4)
        
    def genrawdir(self) -> list[str]:
        """Creates the raw data directory as requires by nnU-Net V2

        Returns:
            list(typing.Union[str, os.PathLike]): list of relevant main subdirs
        """
        self._divtesttrain()
        self._structurerawdir()
        self._senddatatorawdir()
        self._gendatasetjson()
        
        logger.info(f'Dataset created succesfully at {self.datasetdir}')
        # Returns the raw, preprocessed and results directories
        return self.subdirs