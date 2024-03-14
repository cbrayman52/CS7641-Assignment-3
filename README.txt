To access the code: go to the following link:
https://github.com/cbrayman52/CS7641-Assignment-3

To run the code, all you have to do is run the submission.py file.

This will perform all experiments and generate all images used in the report.
Images are saved in the 'Images' directory including a subfolder for the specific model being analyzed.
CSVs are saved in the 'Output' directory including a subfolder for the specific model being analyzed.

The Wine Quality Dataset was provided as a csv in the Dataset folder. However it can also be found online here:
https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

Use pip install -r requirements.txt to download the needed libraries.

Libraries used:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score