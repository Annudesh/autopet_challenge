import os
from .utils.logger import logger

class WorkFlow:
    """ Class to handle data preparation and nnunet training
    """
    def __init__(
        self,
        datadir : str = '/data/blobfuse/FDG-PET-CT-Lesions',
        maindir : str = '/data/blobfuse/maindir'
    ) -> None:
        """__init__ Constructor

        Args:
            datadir (str, optional): Path to data directory supplied by challenge. Defaults to '/data/blobfuse/FDG-PET-CT-Lesions'.
            maindir (str, optional): Path to directory where all intermediate and results files are saved. Defaults to '/data/blobfuse/maindir'.
        """
        self.datadir = datadir
        self.maindir = maindir
        #os.makedirs(self.maindir)
        
        
    
    def datacompile(self)->None:
        """datacompile Compile the provided input data
        """
        from .utils.datacompile import CompileData
        datacompiler = CompileData(datadir=self.datadir,maindir=self.maindir)
        
        self.rawdir, self.preprocesseddir, self.resultsdir = datacompiler.genrawdir()
        
        logger.info('Compiled input data directory...')
        
        # from .utils.constants import ENVKEYS
        # os.environ[ENVKEYS.raw]=self.rawdir
        # os.environ[ENVKEYS.preprocessed]=self.preprocesseddir
        # os.environ[ENVKEYS.results]=self.resultsdir
        
    def nnunetv2(self)->None:
        """nnunetv2 Run nnunetv2 workflow
        """
        from .utils.constants import ENVKEYS
        os.environ[ENVKEYS.raw]="/data/blobfuse/maindir/raw"
        os.environ[ENVKEYS.preprocessed]="/data/blobfuse/maindir/preprocessed"
        os.environ[ENVKEYS.results]="/data/blobfuse/maindir/results"
        os.environ['OMP_NUM_THREADS']='8'
        os.environ['nnUNet_n_proc_DA']='10'
        
        logger.info('Running nnunetv2 workflow...')
        
        from .utils.nnunetv2 import Runner
        nnunetv2runner = Runner(maindir=self.maindir)
        nnunetv2runner.run()
        
    def run(self)->None:
        """run run whole nnunetv2 workflow
        """
        self.datacompile()
        self.nnunetv2()
        
        
        