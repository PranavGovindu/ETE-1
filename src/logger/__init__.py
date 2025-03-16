import logging
import os
from datetime import datetime
logger=logging.getLogger()
logger.setLevel('DEBUG')

root_dir=os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__),'../')))
log_dir_path=os.path.join(root_dir,'logs')
os.makedirs(log_dir_path,exist_ok=True)
log_file_path=os.path.join(f"{ datetime.now().strftime("%m_%d_%y_%H_%M_%S") }.log")

def configure_logger():
    """
    Configure logger
    """
    logger=logging.getLogger()
    
    logger.setLevel('DEBUG')
    
    formatt=logging.Formatter("f[%(asctime)s ] %(name)s -  %(levelname)s - %(message)s")
    
    file_handler=logging.FileHandler(log_file_path,encoding="utf-8")
    file_handler.setFormatter(formatt)
    
    console_handler=logging.StreamHandler(log_file_path)
    console_handler.setFormatter(formatt)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
configure_logger()