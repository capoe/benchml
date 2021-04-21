"""
for making plots
"""
import numpy as np
import matplotlib.pyplot as plt

from asaplib.fit.getscore import get_score
from asaplib.fit import LC_SCOREBOARD

def plot_compare_train_pred(by_model, model_key_now, show=True, save=True, file_prefix='benchml', histogram_bins=30):

    pred_min, pred_max = float("inf"), -float('inf')
    pred_error_min, pred_error_max = float('inf'), -float('inf')

    fig, ax = plt.subplots(ncols=2, nrows=len(by_model[model_key_now].keys()),
    		       #sharex=True, sharey=False,
    		       figsize=(5,10))

    for i_now, key_now in enumerate(by_model[model_key_now].keys()):
        model_now = by_model[model_key_now][key_now]
        # train_idcs, y_pred, y_true, submodel_id, train_or_not
        pred_1 = np.asarray(model_now['train'][:,2])
        pred_2 = np.asarray(model_now['train'][:,1])
        #print(get_score(pred_1,pred_2))
        ax[i_now,0].scatter(pred_1, pred_2, s=1, label='train')
        pred_min, pred_max = min(np.amin(pred_1),pred_min), max(np.amax(pred_1),pred_max)
        
        pred_error = pred_1 - pred_2
        n, bins, _ = ax[i_now,1].hist(pred_error, histogram_bins, 
    				  density=True, alpha=0.75, histtype='step', label='train')
        pred_error_min, pred_error_max = min(np.amin(bins),pred_error_min), max(np.amax(bins),pred_error_max)

        pred_1 = np.asarray(model_now['test'][:,2])
        pred_2 = np.asarray(model_now['test'][:,1])
        #print(get_score(pred_1,pred_2))
        ax[i_now,0].scatter(pred_1, pred_2, s=1, label='test')

        pred_error = pred_2 - pred_1
        n, bins, _ = ax[i_now,1].hist(pred_error, histogram_bins, 
    				  density=True, alpha=0.75, histtype='step', label='test')

    for i_now, key_now in enumerate(by_model[model_key_now].keys()):
        ax[i_now,0].set_xlim([pred_min, pred_max])
        ax[i_now,0].set_ylim([pred_min, pred_max])
        xeqx = np.linspace(pred_min, pred_max,50)
        ax[i_now,0].plot(xeqx, xeqx, '--', c='black')
        ax[i_now,0].text(0.7,0.1,
    		     "{:.2f}".format(by_model[model_key_now][key_now]['train_fraction']), 
    		     transform=ax[i_now,0].transAxes,
    		    bbox=dict(facecolor='red', alpha=0.1))
        ax[i_now,0].yaxis.set_ticks_position('both')
        ax[i_now,0].set_ylabel("y$_{pred}$")
        
        ax[i_now,1].set_xlim([pred_error_min, pred_error_max])
        ax[i_now,1].yaxis.set_label_position("right")
        ax[i_now,1].yaxis.tick_right()
        ax[i_now,1].yaxis.set_ticks_position('both')
        ax[i_now,1].set_ylabel("frequency")
        
    ax[0,0].legend(loc="upper left", bbox_to_anchor=(0.0, 1.9))
    ax[0,1].legend(loc="upper right", bbox_to_anchor=(1.0, 1.9))
    #ax[0,0].text(1.1, 1.1, 'matplotlib', horizontalalignment='center',
    #             verticalalignment='center', transform=ax[0,0].transAxes)

    ax[-1,0].set_xlabel("y$_{true}$")
    ax[-1,1].set_xlabel("y$_{pred}$-y$_{true}$")

    fig.suptitle(file_prefix, fontsize=11, y=0.96)

    plt.tight_layout()
    if show: plt.show()

    if save: fig.savefig(file_prefix+model_key_now+'.pdf')

    return fig, ax

def get_learning_curve(by_model, model_key_now='bmol_soap_rr_00', sc_name='RMSE', ax=None, linecolor='black', label=None):
    
    lc_train_scores = LC_SCOREBOARD([ len(v[1]['train'])/v[1]['n_repeats'] for v in by_model[model_key_now].items() ])
    lc_scores = LC_SCOREBOARD([ len(v[1]['train'])/v[1]['n_repeats'] for v in by_model[model_key_now].items() ])

    for i_now, key_now in enumerate(by_model[model_key_now].keys()):
        model_now = by_model[model_key_now][key_now]
        n_repeats = model_now['n_repeats']
        n_train = len(model_now['train'])/model_now['n_repeats']
        for n_now in range(n_repeats):
            # train_idcs, y_pred, y_true, submodel_id, train_or_not
            sub_model_train = model_now['train'][model_now['train'][:,-2]==n_now][:,[1,2]]
            #print(n_now, np.shape(sub_model_train))
            sub_model_test = model_now['test'][model_now['test'][:,-2]==n_now][:,[1,2]]
            #print(n_now, np.shape(sub_model_test))
        
            lc_score_now = get_score(np.asarray(sub_model_train[:,0]), np.asarray(sub_model_train[:,1]))
            lc_train_scores.add_score(n_train, lc_score_now)
    
            lc_score_now = get_score(np.asarray(sub_model_test[:,0]), np.asarray(sub_model_test[:,1]))
            lc_scores.add_score(n_train, lc_score_now)


    lc_train_results = np.asmatrix(lc_train_scores.fetch(sc_name))
    lc_results = np.asmatrix(lc_scores.fetch(sc_name))

    if ax is not None: 
        ax.errorbar(lc_train_results[:,0], lc_train_results[:,1], yerr=lc_train_results[:,2], 
                    linestyle='--', c = linecolor,
                    uplims=True, lolims=True, label=label)
        ax.errorbar(lc_results[:,0], lc_results[:,1], yerr=lc_results[:,2], 
                    linestyle='-', c = linecolor,
                    uplims=True, lolims=True, label=label)
        if label is not None: ax.legend(loc="right", bbox_to_anchor=(1.3, 0.5))
    
    return lc_results, lc_train_results, ax
