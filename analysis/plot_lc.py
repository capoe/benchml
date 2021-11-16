#! /usr/bin/env python
"""
For plotting learning curves.
e.g.
python plot_lc.py -f benchmark_qm7b.json.gz -p qm7b -c 1 -r 1
"""

import sys
import argparse
import os
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
#import warnings
#warnings.filterwarnings("ignore")
import json

from asaplib.io import NpEncoder, NpDecoder 
from bml_analysis_io import parse
from bml_analysis_func import *
from bml_analysis_plot import *
from bml_model_list import bmol, bxtal, color_dict

def read_acronym(filename="bml_name_acronym.list"):
    # read the acronyms
    acronym_list = np.genfromtxt(filename,dtype=str)
    acronym_dict = {}
    for a in acronym_list:
        acronym_dict[a[0]] = a[1]
    return acronym_dict


def plot_lc_category(model_category, sc_name, best_model_error, lc_by_model, lr_by_model, acronym_dict, color_dict, num_col, num_row):
    if num_row == 0:
        num_row = math.ceil(len(model_category)/num_col)
    fig, ax = plt.subplots(nrows=num_row, ncols=num_col,figsize=(3*num_col,2*num_row),sharex=True,sharey=True)

    for mi, [_, v] in enumerate(dict(sorted(best_model_error.items())).items()):
        i = mi%(num_col*num_row)
        category = v['category']
        best_model = v['model']

        if best_model is None: continue

        if num_row > 1 and num_col > 1: # 2D subplot
            ax_now = ax[i//num_col,i%num_col]
            ax0 = ax[0,0]
        else:
            ax_now = ax[i%(num_col*num_row)]
            ax0 = ax[0]
        
        for key_now in model_category[category]:
            if lc_by_model[key_now] is None: continue
            #print(np.asarray(lc_by_model[key_now])[:,2])
            ax_now.errorbar(np.asarray(lc_by_model[key_now])[:,0], np.asarray(lc_by_model[key_now])[:,1], yerr=np.asarray(lc_by_model[key_now])[:,2],
                    linestyle='-', c=color_dict[category], alpha=0.5,
                    uplims=True, lolims=True)
    
        ax_now.errorbar(np.asarray(lc_by_model[best_model])[:,0], np.asarray(lc_by_model[best_model])[:,1], yerr=np.asarray(lc_by_model[best_model])[:,2],
                linestyle='-',linewidth=4, c=color_dict[category], alpha=1.0, 
                label=acronym_dict[best_model]+"\n"+"LR="+"{:.1e}".format(lr_by_model[best_model]),
                uplims=True, lolims=True)    

        ax_now.legend(loc='best') # bbox_to_anchor=(1.3, 0.5))
        if i//num_col == (num_row -1):
            ax_now.set_xlabel('N',labelpad=-3)
        if i%num_col == 0:
            ax_now.set_ylabel('Test {}'.format(sc_name),labelpad=-2)

    return fig, ax, ax0

def plot_lc_individual(model_category, sc_name, best_model_error, lc_by_model, lr_by_model, acronym_dict, color_dict, num_col, num_row):
    fig, ax = plt.subplots(nrows=num_row, ncols=num_col,figsize=(3*num_col,1*num_row),sharex=True,sharey=True)

    for mi, [_, v] in enumerate(dict(sorted(best_model_error.items())).items()):
        category = v['category']
        best_model = v['model']
        if best_model is None: continue

        for mj, key_now in enumerate(model_category[category]):
            if lc_by_model[key_now] is None: continue
            #print(mj, mi, category, key_now)
            if mj >= num_row:
                print("WARNING: more models in the category " + category + " than the number of rows.")
                continue
            ax_now = ax[mj,mi]
            for key in model_category[category]:
                ax_now.errorbar(np.asarray(lc_by_model[key])[:,0], np.asarray(lc_by_model[key])[:,1], yerr=np.asarray(lc_by_model[key])[:,2],
                    linestyle='-', linewidth=0.4, c=color_dict[category], alpha=0.2,
                    uplims=True, lolims=True)

            ax_now.errorbar(np.asarray(lc_by_model[key_now])[:,0], np.asarray(lc_by_model[key_now])[:,1], yerr=np.asarray(lc_by_model[key_now])[:,2],
                    linestyle='-',linewidth=2, c=color_dict[category], alpha=1.0,
                    label=acronym_dict[key_now]+"\n"+"LR="+"{:.1e}".format(lr_by_model[key_now]),
                    uplims=True, lolims=True)

            ax_now.legend(loc='best') # bbox_to_anchor=(1.3, 0.5))
            if mj == (num_row -1):
                ax_now.set_xlabel('N',labelpad=-3)
            if mi == 0:
                ax_now.set_ylabel('Test {}'.format(sc_name),labelpad=-2)

    return fig, ax, ax[0,0]
    
def plot_lc_together(model_category, sc_name, best_model_error, lc_by_model, lr_by_model, acronym_dict, color_dict):
    fig, ax = plt.subplots(figsize=(8,6))

    for i, [_, v] in enumerate(dict(sorted(best_model_error.items())).items()):
        category = v['category']
        best_model = v['model']
        if best_model is None: continue

        for key_now in model_category[category]:
            if lc_by_model[key_now] is None: continue
            ax.errorbar(np.asarray(lc_by_model[key_now])[:,0], np.asarray(lc_by_model[key_now])[:,1], yerr=np.asarray(lc_by_model[key_now])[:,2],
                    linestyle='-', c=color_dict[category], alpha=0.5,
                    uplims=True, lolims=True)

        ax.errorbar(np.asarray(lc_by_model[best_model])[:,0], np.asarray(lc_by_model[best_model])[:,1], yerr=np.asarray(lc_by_model[best_model])[:,2],
                linestyle='-',linewidth=4, c=color_dict[category], alpha=1.0,
                label=acronym_dict[best_model]+" LR="+"{:.1e}".format(lr_by_model[best_model]),
                uplims=True, lolims=True)

    ax.legend(loc='best') # bbox_to_anchor=(1.3, 0.5))
    ax.set_xlabel('N',labelpad=-10)
    ax.set_ylabel('Test {}'.format(sc_name),labelpad=-5)
    return fig, ax, ax

def main(filename, prefix, plot_type, sc_name, ncol, nrow):

    lc_by_model = {}
    #lc_by_model_train = {}
    lr_by_model = {}

    best_error = 10**20. # just for setting the range of the plot
    worst_error = 0.0

    sorted_model_category = {}
    best_model_error = {}

    read_lc = os.path.exists(prefix+'-lc.json')
    if read_lc:
        with open(prefix+'-lc.json', 'r') as jd:
            lc_by_model = json.load(jd, cls=NpDecoder)
        print("load LCs from json file.")
        all_model_keys = list(lc_by_model.keys())
    else:
        by_model = parse(filename, verbose=True)
        all_model_keys = list(by_model.keys())

    print(all_model_keys)
    acronym_dict = read_acronym()

    if any(all_model_keys[0] in subl for subl in list(bmol.values())):
        model_category = bmol
    elif any(all_model_keys[0] in subl for subl in list(bxtal.values())):
        model_category = bxtal
    else:
        raise ValueError("unknown model type.")

    # compute the learning curves and sort the models in each catagory
    for mi, [category, model_key] in enumerate(model_category.items()):
        sorted_model_category[category] = {}

        for key_now in model_key:
            try:
                if not read_lc:
                    lc_by_model[key_now], _, _ = get_learning_curve(by_model, model_key_now=key_now,
                                                       sc_name=sc_name)
                    #lc_by_model[key_now], lc_by_model_train[key_now], _ = get_learning_curve(by_model, model_key_now=key_now,
                    #                                       sc_name=sc_name)
                else:
                    lc_by_model[key_now] = np.asarray(lc_by_model[key_now])
                sorted_model_category[category][key_now] = np.min(lc_by_model[key_now][:,1])

                if np.min(lc_by_model[key_now][:,1]) < best_error:
                    best_error = np.min(lc_by_model[key_now][:,1])
                if lc_by_model[key_now][0,1] > worst_error:
                    worst_error = lc_by_model[key_now][0,1]
                N_range =  lc_by_model[key_now][-1,0]

                # compute learning rate
                lc_now = lc_by_model[key_now]
                lr_by_model[key_now] = -(np.log(lc_now[0,1])-np.log(lc_now[-1,1]))/(np.log(lc_now[0,0])-np.log(lc_now[-1,0]))
            except:
                lc_by_model[key_now] = None
                continue
        #print(sorted_model_category[category])

        try:
            # sort the models in each category
            sorted_model_category[category] = dict(sorted(sorted_model_category[category].items(), key=lambda item: item[1]))
            # print(sorted_model_category[category])
            [ best_in_catogory, best_in_catogory_error ] = next(iter( sorted_model_category[category].items() ))
            best_model_error[best_in_catogory_error] = { "category": category, "model": best_in_catogory}
        except:
            continue


    # dumpy the data for the learning curves and learning rates
    if not read_lc:
        test_RMSE = [ [k, lc_by_model[k][-1,1]] for k in by_model.keys() if lc_by_model[key_now] is not None]
        np.savetxt(prefix+'-test_RMSE.dat', test_RMSE, fmt='%s')
        with open(prefix+'-lc.json', 'w') as jd:
            json.dump(lc_by_model,jd, sort_keys=False, cls=NpEncoder)

    # plot
    if plot_type == 'together':
        fig, ax, ax0 = plot_lc_together(sorted_model_category, sc_name, best_model_error, lc_by_model, lr_by_model, acronym_dict, color_dict)
    elif plot_type == 'category':
        fig, ax, ax0 = plot_lc_category(sorted_model_category, sc_name, best_model_error, lc_by_model, lr_by_model, acronym_dict, color_dict, ncol, nrow)
    elif plot_type == 'individual':
        fig, ax, ax0 = plot_lc_individual(sorted_model_category, sc_name, best_model_error, lc_by_model, lr_by_model, acronym_dict, color_dict, ncol, nrow)
    else:
        raise ValueError("plot type not recognized.")

    if (worst_error/best_error) > 10:
        ax0.set_ylim([best_error*0.7,worst_error*1.3])
        ax0.set_yscale('log')
    else:
        ax0.set_ylim([best_error*0.9,worst_error*1.1])
    if N_range > 100:
        ax0.set_xscale('log')

    ax0.text(0.2, 0.9, prefix, fontsize=14,
         horizontalalignment='left',
         verticalalignment='center',
         transform = ax0.transAxes)
    fig.tight_layout()
    fig.savefig(prefix+'-lc.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-f', '--file', type = str, default = 'bml', help = 'Location of the .json.gz file.')
    parser.add_argument('-p', '--prefix', type = str, default = 'bml', help = 'Prefix of the output file.')
    parser.add_argument('-t', '--type', type = str, default = 'category', help = 'Type for plotting the learning curves: [together/category/individual]')
    parser.add_argument('-m', '--metric', type = str, default = 'RMSE', help = 'The metric for plotting the learning curves.')
    parser.add_argument('-c', '--column', type = int, default = 4, help = 'The number of columns')
    parser.add_argument('-r', '--row', type = int, default = 0, help = 'The number of rows')
    args = parser.parse_args()
    sys.exit(main(args.file, args.prefix, args.type, args.metric, args.column, args.row))

