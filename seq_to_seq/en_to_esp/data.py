import os
import numpy as np
import pandas as pd

path_to_df = os.path.expanduser("~/datasets/en_to_esp/data.csv")
df = pd.read_csv(path_to_df)

df.shape

df.head(10)
