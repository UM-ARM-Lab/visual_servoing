import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

result = pkl.load(open("test-results/20221018-172136/result.pkl", "rb"))

# Get the mocap camera -> eef pose estimates
