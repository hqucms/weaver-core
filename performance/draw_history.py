import matplotlib.pyplot as plt
import os
import re
import pickle
from collections import defaultdict
import mplhep as hep
import yaml
import argparse
import time

'''orig_stdout = sys.stdout
f = open('history.txt', 'w')
sys.stdout = f'''

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--not-show', action='store_true', default=False,
                    help='do not show plots')
parser.add_argument('--save', action='store_true', default=False,
                    help='save plots')
parser.add_argument('--last-epoch', type=int, default=100,
                    help='save plots')
parser.add_argument('--path', type=str, default="",
                    help='input path')
parser.add_argument('--name', type=str, default='',
                    help='name of the configuration')
parser.add_argument('--in-dict', type=str, default='performance_comparison',
                    help='name of the file with the dictionary')
args = parser.parse_args()

# dictionary with the information to extract from the log files
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

def plot(out_dir, name, fig_handle):
    plt.xlabel('Epoch')
    hep.style.use("CMS")
    hep.cms.label(rlabel="")
    hep.cms.lumitext(f'{name}')
    plt.legend()#loc=2, prop={'size': 15})
    plt.grid()
    plt.savefig(f'{out_dir}/history_{name}.png', bbox_inches='tight')
    if args.save:
        with open(f'{out_dir}/{name}.pickle', 'wb') as f:
            pickle.dump(fig_handle, f)
    if not args.not_show:
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
    infile_dict=load_dict(f'{args.in_dict}.yaml')

    # create output directory
    date_time = time.strftime('%Y%m%d-%H%M%S')
    out_dir = os.path.join(f'{args.path}history_plot', f'{date_time}_{args.name}_history')
    os.makedirs(out_dir, exist_ok=True)

    # load history for each input
    for input_name, info in infile_dict.items():
        # get all log files in the input directory and sort them in alphabetical order
        if isinstance(input_name, str):
            dir_name=os.path.join(args.path, input_name)
            infiles = [os.path.join(dir_name,filename) for filename in os.listdir(dir_name) if '.log' in filename]
            infiles.sort()#key=lambda s: int(re.findall(r'\d+', s)[-1]))
        # get specific log files and sort them in alphabetical order
        elif isinstance(input_name, tuple):
            infiles=[os.path.join(args.path+"input", "logs", f"{k}.log") for k in input_name]
            infiles.sort()# key=lambda s: int(re.findall(r'\d+', s)[-1]))
        #print(infiles)
        # read the log files and extract the information
        for infile in infiles:
            with open(infile) as f:
                f = f.readlines()
            # find the line with the information and save it in the dictionary
            for line in f:
                for name, value in history_dict.items():
                    if value[0] in line:
                       info[0][name].append(float(line.split(value[0],1)[1].split(value[1])[0]))

    # plot the history
    for history, _ in history_dict.items():
        fig_handle = plt.figure()
        for _, info in infile_dict.items():
            for name, value in info[0].items():
                if name == history:
                    plt.plot(value[:args.last_epoch+1], info[2], label=f'{name} {info[1]}')
        plot(out_dir, history, fig_handle)





'''
import pickle
figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))

figx.show()
'''