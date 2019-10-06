import os
import pandas as pd

def convert():
    folders = [f'./{x}' for x in os.listdir('./')]
    folders = [x for x in folders if 'good' in x or 'broken' in x]
    
    paths, targets = [], []
    for folder in folders:
        imgs = [os.path.join(os.path.abspath(folder), x) for x in os.listdir(folder)]
        paths.extend(imgs)
        targets.extend([folder.replace('./', '')] * len(imgs))
    frame = pd.DataFrame({
        'path': paths,
        'target' : targets
    })
    frame = frame.sample(frac=1)
    frame.to_csv('converted_data.csv', index=False)

convert()