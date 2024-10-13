import pandas as pd


fp = '/Users/mackenzie/Desktop/Zipper/Fabrication/Wafer11/process-flow_w11.xlsx'
df = pd.read_excel(fp)
dfict = df.to_dict()

pflow = {}
for i in range(len(df)):

    step_dict = {}
    for k, v in dfict.items():
        # step_dict = dict([(k, v[i]) for k, v in dfict.items()])
        v = v[i]
        if k == 'details':
            v = eval(v)
        elif v == 'None':
            v = eval(v)

        step_dict.update({k: v})

    pflow.update({i + 1: step_dict})

j = 2