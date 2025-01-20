import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dir',  required=True)

    args = parser.parse_args()

    df = pd.read_csv(os.path.join(args.dir, 'progress.csv'))

    param ='evaluation/Average Returns' 
    y = df[param]
    
    plt.plot(y)
    plt.xlabel("Epoch")
    plt.ylabel(param)
    plt.savefig(args.dir)