import sys
sys.path.append('../')
from src.config import *
import os
from os import listdir
from os.path import isfile,join

def list_all_dataset():
    folders= os.listdir(os.path.join(UPLOAD_FOLDER))
    #load the models to memory only once, when the app boots
    for folder in folders:
        folder = {
            "dataset": folder,
            "success": True
        }
        folders.append(folder)
    return folders