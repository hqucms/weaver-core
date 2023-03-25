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
from scipy.ndimage import uniform_filter1d

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--not-show', action='store_true', default=False,
                    help='do not show plots')
parser.add_argument('--save', action='store_true', default=False,
                    help='save plots')
parser.add_argument('--not-partial', action='store_true', default=False,
                    help='ignore partial epochs in the plots')
parser.add_argument('--last-epoch', type=int, default=14,
                    help='save plots')
parser.add_argument('--in-path', type=str, default='',
                    help='input path')
parser.add_argument('--out-path', type=str, default='',
                    help='output path')
parser.add_argument('--in-dict', type=str, default='total',
                    help='input dictionary')
parser.add_argument('--complete-dict', type=str, default='complete_dict',
                    help='dictionary with names')
parser.add_argument('--name', type=str, default='',
                    help='name of the configuration')
parser.add_argument('--type', type=str, default='lite,full',
                    help='name of the file with the dictionary')
parser.add_argument('--num-partial', type=int, default=3,
                    help='number of partial samplings per epoch')
parser.add_argument('--history-config', type=str, default='history_config',
                    help='name of the file with the dictionary')
args = parser.parse_args()

# type of the network
if ',' in args.type:
    NET_TYPES = [k for k in args.type.split(',')]
else:
    NET_TYPES = [args.type]

with open(f'{args.history_config}.yaml', 'r') as stream:
    history_dict=yaml.safe_load(stream)['history_dict']

def plot(out_dir, name, fig_handle):
    """Plot the history of the given figure handle
    and save it in the given output directory

    :param out_dir: output directory
    :param name: name of the plot
    :param fig_handle: figure handle
    """
    plt.xlabel('Epoch', fontsize=20, loc='right')
    hep.style.use("CMS")
    hep.cms.label('Preliminary')
    hep.cms.label(year='UL18')
    #plt.suptitle(name, horizontalalignment='center', verticalalignment='top', fontsize=25)

    plt.legend(labelcolor='linecolor')
    plt.grid()
    plt.savefig(f'{out_dir}/history_{name}.png', bbox_inches='tight')
    if args.save:
        with open(f'{out_dir}/history_{name}.pickle', 'wb') as f:
            pickle.dump(fig_handle, f)
    if not args.not_show:
        plt.show()
    plt.close()

def load_dict(complete_dict, in_dict):
    """Load the dictionary with the information to plot
    :param    complete_dict : string with the name of the file containing the paths to the models
    :param    in_dict : string with the name of the file containing the names of the models to load
    :return   info_dict : dictionary with the models to load
    """
    with open(complete_dict, 'r') as stream:
        loaded_dict=yaml.safe_load(stream)
    with open(in_dict, 'r') as stream:
        in_names=yaml.safe_load(stream)['networks']
    info_dict = defaultdict(list)
    for k, v in loaded_dict.items():
        if v[0] not in in_names:
            continue
        #dictionary with the path, the name of the model and the color
        info_dict[k].append(defaultdict(list))
        info_dict[k].append(v[0])
        info_dict[k].append(v[1])
    return info_dict

def draw_plot(value, num_tot, name, info, save):
    if len(value[:num_tot]) == num_tot:
        x_part = np.linspace(-(args.num_partial-1)/args.num_partial, 0, args.num_partial)
        x = np.concatenate([x_part + i for i in range(args.last_epoch+1)])
        #y=value[:num_tot]
        y=uniform_filter1d(value[:num_tot], size=args.num_partial)
    else:
        x = np.linspace(0, len(value[:args.last_epoch]), len(value[:args.last_epoch+1]))
        y=value[:args.last_epoch+1]
    plt.plot(x, y, color=info[2], label=f'{name} {info[1]}')
    save=True

    return save

if __name__ == "__main__":

    tot_dict = {}
    # create output directory
    date_time = time.strftime('%Y%m%d-%H%M%S')
    main_out_dir = os.path.join(f'{args.out_path}history_plot', f'{date_time}_{args.name}_{args.in_dict}_{args.type}_history')
    os.makedirs(main_out_dir, exist_ok=True)
    print(main_out_dir)

    for net_type in NET_TYPES:
        infile_dict=load_dict(f'config/{args.complete_dict}.yaml', f'config/{args.in_dict}_{net_type}.yaml')
        out_dir = os.path.join(main_out_dir, f'{args.name}_{args.in_dict}_{net_type}_history')
        os.makedirs(out_dir, exist_ok=True)

        # load history for each input
        for input_name, info in infile_dict.items():
            # get all log files in the input directory and sort them in alphabetical order
            if isinstance(input_name, str):
                dir_name=os.path.join(args.in_path, input_name)
                infiles = [os.path.join(dir_name,filename) for filename in os.listdir(dir_name) if '.log' in filename]
                infiles.sort()#key=lambda s: int(re.findall(r'\d+', s)[-1]))
            # get specific log files and sort them in alphabetical order
            elif isinstance(input_name, list):
                infiles=[os.path.join(args.in_path+"input", "logs", f"{k}.log") for k in input_name]
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
                            if args.not_partial and 'Partial' in line:
                                continue
                            val=float(line.split(value[0],1)[1].split(value[1])[0])
                            if val > 100:
                                val=-1
                            info[0][name].append(val)
        tot_dict[net_type]=infile_dict


        # plot the history
        num_tot=(args.last_epoch+1)*args.num_partial
        for history, _ in history_dict.items():
            fig_handle = plt.figure(figsize=(30, 20))
            save=False
            for _, info in infile_dict.items():
                for name, value in info[0].items():
                    if name == history and any(val != 0 for val in value):
                        save=draw_plot(value, num_tot, name, info, save)
            # call function plot only if figure is not empty
            if save:
                plot(out_dir, f'{history}_{args.in_dict}_{net_type}', fig_handle)
            else:
                plt.close()

    if len(NET_TYPES) > 1:
        infile_dict=load_dict(f'config/{args.complete_dict}.yaml', f'config/{args.in_dict}_{NET_TYPES[0]}.yaml')
        out_dir = os.path.join(main_out_dir, f'{args.name}_{args.in_dict}_{args.type}_history')
        os.makedirs(out_dir, exist_ok=True)

        # plot the history
        num_tot=(args.last_epoch+1)*args.num_partial
        for history, _ in history_dict.items():
            fig_handle = plt.figure(figsize=(30, 20))
            save=False
            for net_type in NET_TYPES:
                for _, info in tot_dict[net_type].items():
                    for name, value in info[0].items():
                        if name == history and any(val != 0 for val in value):
                            save=draw_plot(value, num_tot, name, info, save)
            # call function plot only if figure is not empty
            if save:
                plot(out_dir, f'{history}_{args.in_dict}_{args.type}', fig_handle)
            else:
                plt.close()




'''
import pickle
figx = pickle.load(open('FigureObject.fig.pickle', 'rb'))

figx.show()
'''