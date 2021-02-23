import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd

train_data_dir = "../DL_ASS_1/train/"
test_data_dir  = "../DL_ASS_1/test/"

def count_exp(path, set_):
    df = None
    dict_ = {}
    for expression in os.listdir(path):
        if not expression.startswith('.'):
            dir_ = path + expression
            dict_[expression] = len(os.listdir(dir_))
            df = pd.DataFrame(dict_, index=[set_])
    return df

train_count = count_exp(train_data_dir, 'train')
test_count  = count_exp(test_data_dir, 'test')

total_count = (train_count, test_count)
total_count = pd.concat(total_count)
total_count = total_count.sum()
print(total_count)
# total_count = np.stack(train_count, test_count)

train_count.transpose().plot(kind = 'bar', legend=None, ylabel='Frequency', xlabel='Classes', title='Class distribution', rot=0.)
plt.savefig(fname='class_dist', format='pdf')

