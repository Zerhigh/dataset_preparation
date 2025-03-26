# execute this script before a train test split to modiFy the stratiffication to remove unneccessary image containing non classes

import pathlib
from pathlib import Path
import pandas as pd


BASE = Path("C:/Users/PC/Desktop/TU/Master/MasterThesis/data/metadata/")
meta = pd.read_csv(BASE / "final_taco_metadata.csv")


pass