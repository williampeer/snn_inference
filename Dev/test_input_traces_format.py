import pandas as pd
inp_trace = pd.read_csv('input_traces_hh.csv', index_col=0).to_numpy()
print('inp_trace:', inp_trace)
