import pandas as pd
import matplotlib.pyplot as plt

from graycart.utils import process


d_fn = '/Users/mackenzie/Desktop/Zipper/Fabrication/Wafer18/results/w18_merged_process_profiles.xlsx'
dff = pd.read_excel(d_fn)

steps = [4, 5]
fid = 0