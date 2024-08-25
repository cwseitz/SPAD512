import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from skimage.io import imsave, imread
from scipy.signal import convolve, deconvolve
from concurrent.futures import ProcessPoolExecutor, as_completed
import random

@staticmethod
def plotLifetimes(mean_image, std_image, widths, steps, tau, savename, show=True):
    numtau, xlen, ylen = np.shape(mean_image)
    
    mean_image[np.isnan(mean_image)] = -1
    std_image[np.isnan(std_image)] = -1

    steps = np.asarray(steps) * 1e-3
    widths = np.asarray(widths) * 1e-3

    # plot mean
    if (numtau == 1):
        fig, ax = plt.subplots(1,2,figsize=(12,6))

        lower = min(max(tau[0] - 5, int(np.min(mean_image[0]))), tau[0] - 1)
        upper = max(min(tau[0] + 5, int(np.max(mean_image[0] + 1))), tau[0] + 1)
        norm = mcolors.TwoSlopeNorm(vmin=lower, vcenter=tau[0], vmax=upper)
        cax1 = ax[0].imshow(mean_image[0], cmap='seismic', norm=norm)
        cbar1 = fig.colorbar(cax1, ax=ax[0], shrink = 0.6)
        cbar1.set_label('Means, ns')
        ax[0].set_title('Mean Lifetimes')
        ax[0].set_xlabel('Step size (ns)')
        ax[0].set_ylabel('Widths (ns)')
        ax[0].set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
        ax[0].set_yticklabels(widths)
        ax[0].set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
        ax[0].set_xticklabels(steps)
        plt.setp(ax[0].get_xticklabels(), rotation=45)

        # plot stdevs
        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2)
        cax2 = ax[1].imshow(std_image[0], cmap='seismic', norm=norm)
        cbar2 = fig.colorbar(cax2, ax=ax[1], shrink = 0.6)
        cbar2.set_label('St Devs, ns')
        ax[1].set_title('Standard Deviation of Lifetimes')
        ax[1].set_xlabel('Step size (ns)')
        ax[1].set_ylabel('Widths (ns)')
        ax[1].set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
        ax[1].set_yticklabels(widths)
        ax[1].set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
        ax[1].set_xticklabels(steps)
        plt.setp(ax[1].get_xticklabels(), rotation=45)

        plt.savefig(savename, bbox_inches='tight')
        print('Figure saved as ' + savename)

        if show:
            plt.show()

    if (numtau == 2):
        fig, ax = plt.subplots(2,2,figsize=(12,12))

        if (np.mean(mean_image[0]) < np.mean(mean_image[1])):
            temp = mean_image[1].copy()
            mean_image[1] = mean_image[0]
            mean_image[0] = temp

            temp = std_image[1].copy()
            std_image[1] = std_image[0]
            std_image[0] = temp

        # temp = tau[1]
        # tau[1] = tau[0]
        # tau[0] = temp

        lower = min(max(tau[0] - 10, int(np.min(mean_image[0]))), tau[0] - 1)
        upper = max(min(tau[0] + 10, int(np.max(mean_image[0] + 1))), tau[0] + 1)
        norm = mcolors.TwoSlopeNorm(vmin=lower, vcenter=tau[0], vmax=upper)
        cax1 = ax[0, 0].imshow(mean_image[0], cmap='seismic', norm=norm)
        cbar1 = fig.colorbar(cax1, ax=ax[0, 0], shrink = 0.6)
        cbar1.set_label('Means, ns')
        ax[0, 0].set_title('Larger Lifetimes')
        ax[0, 0].set_xlabel('Step size (ns)')
        ax[0, 0].set_ylabel('Integration (ms)')
        ax[0, 0].set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
        ax[0, 0].set_yticklabels(np.round(widths, 2))
        ax[0, 0].set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
        ax[0, 0].set_xticklabels(np.round(steps,2))
        plt.setp(ax[0, 0].get_xticklabels(), rotation=45)

        lower = min(max(tau[1] - 5, int(np.min(mean_image[1]))), tau[1] - 1)
        upper = max(min(tau[1] + 5, int(np.max(mean_image[1] + 1))), tau[1] + 1)
        norm = mcolors.TwoSlopeNorm(vmin=lower, vcenter=tau[1], vmax=upper)
        cax2 = ax[0, 1].imshow(mean_image[1], cmap='seismic', norm=norm)
        cbar2 = fig.colorbar(cax2, ax=ax[0, 1], shrink = 0.6)
        cbar2.set_label('Means, ns')
        ax[0, 1].set_title('Smaller Lifetimes')
        ax[0, 1].set_xlabel('Step size (ns)')
        ax[0, 1].set_ylabel('Integration (ms)')
        ax[0, 1].set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
        ax[0, 1].set_yticklabels(np.round(widths, 2))
        ax[0, 1].set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
        ax[0, 1].set_xticklabels(np.round(steps,2))
        plt.setp(ax[0, 1].get_xticklabels(), rotation=45)

        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=10)
        cax3 = ax[1, 0].imshow(std_image[0], cmap='seismic', norm=norm)
        cbar3 = fig.colorbar(cax3, ax=ax[1, 0], shrink = 0.6)
        cbar3.set_label('St Devs, ns')
        ax[1, 0].set_title('Standard Deviation of Lifetimes')
        ax[1, 0].set_xlabel('Step size (ns)')
        ax[1, 0].set_ylabel('Integration (ms)')
        ax[1, 0].set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
        ax[1, 0].set_yticklabels(np.round(widths, 2))
        ax[1, 0].set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
        ax[1, 0].set_xticklabels(np.round(steps,2))
        plt.setp(ax[1, 0].get_xticklabels(), rotation=45)

        norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=10)
        cax3 = ax[1, 1].imshow(std_image[1], cmap='seismic', norm=norm)
        cbar3 = fig.colorbar(cax3, ax=ax[1, 1], shrink = 0.6)
        cbar3.set_label('St Devs, ns')
        ax[1, 1].set_title('Standard Deviation of Lifetimes')
        ax[1, 1].set_xlabel('Step size (ns)')
        ax[1, 1].set_ylabel('Integration (ms)')
        ax[1, 1].set_yticks(np.linspace(0, xlen, num=xlen, endpoint=False))
        ax[1, 1].set_yticklabels(np.round(widths, 2))
        ax[1, 1].set_xticks(np.linspace(0, ylen, num=ylen, endpoint=False))
        ax[1, 1].set_xticklabels(np.round(steps,2))
        plt.setp(ax[1, 1].get_xticklabels(), rotation=45)

        plt.savefig(savename, bbox_inches='tight')
        print('Figure saved as ' + savename)

        if show:
            plt.show()