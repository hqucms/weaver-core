import matplotlib.pyplot as plt
import os
import re
import pickle
from collections import defaultdict
import mplhep as hep
import yaml
import argparse
import time
import numpy as np

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--not-show', action='store_true', default=False,
                    help='do not show plots')
parser.add_argument('--save', action='store_true', default=False,
                    help='save plots')
parser.add_argument('--not-partial', action='store_true', default=False,
                    help='ignore partial epochs in the plots')
parser.add_argument('--last-epoch', type=int, default=100,
                    help='save plots')
parser.add_argument('--in-path', type=str, default='',
                    help='input path')
parser.add_argument('--out-path', type=str, default='',
                    help='output path')
parser.add_argument('--name', type=str, default='',
                    help='name of the configuration')
parser.add_argument('--type', type=str, default='',
                    help='name of the file with the dictionary')
args = parser.parse_args()

# type of the network
if not args.type:
    NET_TYPES = ['lite', 'full']
else:
    NET_TYPES = [args.type]

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
    """Plot the history of the given figure handle
    and save it in the given output directory

    :param out_dir: output directory
    :param name: name of the plot
    :param fig_handle: figure handle
    """
    plt.xlabel('Epoch')
    hep.style.use("CMS")
    hep.cms.label(rlabel="")
    plt.suptitle(name, horizontalalignment='center', verticalalignment='top', fontsize=25)

    #hep.cms.lumitext(name)
    plt.legend(labelcolor='linecolor')#loc=2, prop={'size': 15})
    plt.grid()
    plt.savefig(f'{out_dir}/history_{name}.png', bbox_inches='tight')
    if args.save:
        with open(f'{out_dir}/history_{name}.pickle', 'wb') as f:
            pickle.dump(fig_handle, f)
    if not args.not_show:
        plt.show()
    plt.close()

def load_dict(name):
    """Load the dictionary with the information to plot
    :param name: name of the dictionary
    :return: dictionary with the information to plot"""
    with open(name, 'r') as stream:
        loaded_dict=yaml.safe_load(stream)
    info_dict = defaultdict(list)
    for k, v in loaded_dict.items():
        info_dict[k].append(defaultdict(list))
        info_dict[k].append(v[0])
        info_dict[k].append(v[1])
    return info_dict

if __name__ == "__main__":

    # create output directory
    date_time = time.strftime('%Y%m%d-%H%M%S')
    main_out_dir = os.path.join(f'{args.out_path}history_plot', f'{date_time}_{args.name}_history')
    os.makedirs(main_out_dir, exist_ok=True)

    for net_type in NET_TYPES:
        infile_dict=load_dict(f'performance_comparison_{net_type}.yaml')
        out_dir = os.path.join(main_out_dir, f'{args.name}_history_{net_type}')
        os.makedirs(out_dir, exist_ok=True)

        # load history for each input
        for input_name, info in infile_dict.items():
            # get all log files in the input directory and sort them in alphabetical order
            if isinstance(input_name, str):
                dir_name=os.path.join(args.in_path, input_name)
                infiles = [os.path.join(dir_name,filename) for filename in os.listdir(dir_name) if '.log' in filename]
                infiles.sort()#key=lambda s: int(re.findall(r'\d+', s)[-1]))
            # get specific log files and sort them in alphabetical order
            elif isinstance(input_name, tuple):
                infiles=[os.path.join(args.in_path+"input", "logs", f"{k}.log") for k in input_name]
                infiles.sort()# key=lambda s: int(re.findall(r'\d+', s)[-1]))
            print(infiles)
            # read the log files and extract the information
            for infile in infiles:
                with open(infile) as f:
                    f = f.readlines()
                # find the line with the information and save it in the dictionary
                for line in f:
                    for name, value in history_dict.items():
                        if value[0] in line:
                            if args.not_partial and 'Partial' in line:
                                continue
                            info[0][name].append(float(line.split(value[0],1)[1].split(value[1])[0]))

        # plot the history
        for history, _ in history_dict.items():
            fig_handle = plt.figure(figsize=(30, 20))
            for _, info in infile_dict.items():
                for name, value in info[0].items():
                    if name == history and any(val != 0 for val in value):
                        x = np.linspace(0, len(value), len(value)) if args.not_partial else np.linspace(0, len(value), len(value)*3)
                        plt.plot(x, value[:args.last_epoch+1], info[2], label=f'{name} {info[1]}')
            plot(out_dir, history, fig_handle)





'''
import pickle
figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))

figx.show()
'''