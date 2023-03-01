import matplotlib.pyplot as plt
import sys
import os
import re
import pickle
from collections import defaultdict
import mplhep as hep
import yaml
import argparse

hep.style.use("CMS")

'''orig_stdout = sys.stdout
f = open('history.txt', 'w')
sys.stdout = f'''


parser = argparse.ArgumentParser()
parser.add_argument('--show', action='store_false', default=True,
                    help='show plots')
parser.add_argument('--save', action='store_true', default=False,
                    help='save plots')
parser.add_argument('--last-epoch', type=int, default=100,
                    help='save plots')
args = parser.parse_args()


history_dict = {
    'primary_train_loss': ('AvgLoss: ', ','),
    'primary_train_acc': ('AvgAcc: ', '\n'),
    'comb_train_loss': ('AvgCombLoss: ', ','),
    'aux_train_loss': ('AvgAuxLoss: ', ','),
    'aux_train_accPF': ('AvgAuxAccPF: ', ','),
    'aux_train_dist': ('AvgAuxDist: ', ','),
    'aux_train_accPair': ('AvgAuxAccPair: ', '\n'),
    'primary_val_metric': ('validation metric: ', ' ('),
    'primary_val_loss': ('validation loss: ', ' ('),
    'comb_val_loss': ('validation combined loss: ', ' ('),
    'aux_val_metricPF': ('validation aux metric PF: ', ' ('),
    'aux_val_dist': ('validation aux distance: ', ' ('),
    'aux_val_metricPair': ('validation aux metric pair: ', ' ('),
    'aux_val_loss': ('validation aux loss: ', ' ('),
    'primary_test_metric': ('test metric: ', ' ('),
    'primary_test_loss': ('test loss: ', ' ('),
    'comb_test_loss': ('test combined loss: ', ' ('),
    'aux_test_metricPF': ('test aux metric PF: ', ' ('),
    'aux_test_dist': ('test aux distance: ', ' ('),
    'aux_test_metricPair': ('test aux metric pair: ', ' ('),
    'aux_test_loss': ('test aux loss: ', ' ('),
}

def plot(name, fig_handle):
    plt.xlabel('Epoch')
    hep.cms.label(rlabel="")
    hep.cms.lumitext(f'{name}')
    plt.legend()#loc=2, prop={'size': 15})
    plt.savefig(f'history_plot/history_{name}.png', bbox_inches='tight')
    if args.save:
        with open(f'history_plot/{name}.pickle', 'wb') as f:
            pickle.dump(fig_handle, f)
    if args.show:
        plt.show()

def load_dict(name):
    with open(name, 'r') as stream:
        loaded_dict=yaml.safe_load(stream)
    info_dict = defaultdict(list)
    for k, v in loaded_dict.items():
        info_dict[k].append(defaultdict(list))
        info_dict[k].append(v[0])
        info_dict[k].append(v[1])
    return info_dict

if __name__ == "__main__":
    infile_dict=load_dict('performance_comparison.yaml')
    for input_name, info in infile_dict.items():
        if isinstance(input_name, str):
            dir_name=os.path.join("input", input_name)
            infiles = [os.path.join(dir_name,filename) for filename in os.listdir(dir_name) if '.log' in filename]
            #print(infiles)
            infiles.sort(key=lambda s: int(re.findall(r'\d+', s)[-1]))
        elif isinstance(input_name, tuple):
            infiles=[os.path.join("input", "logs", f"{k}.log") for k in input_name]
            infiles.sort(key=lambda s: int(re.findall(r'\d+', s)[-1]))
        #print(infiles)
        for infile in infiles:
            with open(infile) as f:
                f = f.readlines()

            for line in f:
                for name, value in history_dict.items():
                    if value[0] in line:
                       info[0][name].append(float(line.split(value[0],1)[1].split(value[1])[0]))

        #print(f'loss {info[1]}:    ', info[0])


    for history, _ in history_dict.items():
        fig_handle = plt.figure()
        for _, info in infile_dict.items():
            for name, value in info[0].items():
                if name == history:
                    plt.plot(value[:args.last_epoch+1], info[2], label=f'{name} {info[1]}')
        plot(history, fig_handle)





'''
import pickle
figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))

figx.show()
'''