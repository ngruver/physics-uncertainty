import sys
import numpy as np
import scipy.stats as st

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.lines import Line2D
import seaborn as sns

def viz_static_body(pred_zs, true_zs, body_idx, color='r'):
    all_zs = np.reshape(pred_zs, (-1, *np.shape(pred_zs)[2:]))
    pred_mean = np.mean(pred_zs, axis=0)

    # sns.kdeplot(x=all_zs[:,0], y=all_zs[:,1], fill=True, color=color, alpha=0.5)
    plt.scatter(all_zs[:,0], all_zs[:,1], c=color, alpha=0.06) 
    plt.scatter(true_zs[0,0], true_zs[0,1], c=color)
    plt.plot(pred_mean[:,0], pred_mean[:,1], c=color, linestyle='--')
    plt.plot(true_zs[:,0], true_zs[:,1], c=color)

    legend_objs = []
    legend_objs.append(Line2D([0], [0], marker='o', markerfacecolor=color, 
                                        color='w', label='body {} init cond'.format(body_idx)))
    legend_objs.append(Line2D([0], [0], color=color, label='body {} ground truth'.format(body_idx)))
    legend_objs.append(Line2D([0], [0], color=color, linestyle='--',
                                        label='body {} pred mean'.format(body_idx)))
    legend_objs.append(mpatches.Patch(color=color, alpha=0.5,
                                      label='body {} pred dist'.format(body_idx)))
    return legend_objs

def viz_static(pred_zs, true_zs):
    print(np.shape(pred_zs))
    print(np.shape(true_zs))

    num_samples = np.shape(true_zs)[0]
    num_bodies = np.shape(true_zs)[-2]
    colors = ['r','b','g']
    colors = colors[:num_bodies]

    for sample_idx in range(num_samples):
        legend_objs = []
        for body_idx in range(num_bodies):
            legend_objs += viz_static_body(pred_zs[:,sample_idx,:,body_idx,:],
                                           true_zs[sample_idx,:,body_idx,:],
                                           body_idx+1, colors[body_idx])
        plt.legend(handles=legend_objs)
        plt.show()

        break

def viz_dynamic_body(pred_zs, true_zs, body_idx, color='r'):
    all_zs = np.reshape(pred_zs, (-1, *np.shape(pred_zs)[2:]))
    pred_mean = np.mean(pred_zs, axis=0)

    sns.kdeplot(x=all_zs[:,0], y=all_zs[:,1], fill=True, color=color, alpha=0.5)
    plt.scatter(true_zs[0,0], true_zs[0,1], c=color)
    plt.plot(pred_mean[:,0], pred_mean[:,1], c=color, linestyle='--')
    plt.plot(true_zs[:,0], true_zs[:,1], c=color)

    # legend_objs = []
    # legend_objs.append(Line2D([0], [0], marker='o', markerfacecolor=color, 
    #                                     color='w', label='body {} init cond'.format(body_idx)))
    # legend_objs.append(Line2D([0], [0], color=color, label='body {} ground truth'.format(body_idx)))
    # legend_objs.append(Line2D([0], [0], color=color, linestyle='--',
    #                                     label='body {} pred mean'.format(body_idx)))
    # legend_objs.append(mpatches.Patch(color=color, alpha=0.5,
    #                                   label='body {} pred dist'.format(body_idx)))
    # return legend_objs

def kde(x, y):
    xmin, xmax = np.amin(x), np.amax(x)
    ymin, ymax = np.amin(y), np.amax(y)

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    return xx, yy, f

def viz_dynamic(pred_zs, true_zs):
    num_samples, num_ts = np.shape(true_zs)[:2]
    num_bodies = np.shape(true_zs)[-2]
    colors = ['r','b','g']
    colors = colors[:num_bodies]

    for sample_idx in range(num_samples):
        fig = plt.figure()
        ax = fig.gca()

        # x_min = np.amin(pred_zs[:,sample_idx,:,:,0])
        # x_max = np.amax(pred_zs[:,sample_idx,:,:,0])
        # y_min = np.amin(pred_zs[:,sample_idx,:,:,1])
        # y_max = np.amax(pred_zs[:,sample_idx,:,:,1])

        # ax.set_xlim((x_min, x_max))
        # ax.set_ylim((y_min, y_max))

        for body_idx in range(num_bodies):
            ax.plot(true_zs[sample_idx,:,body_idx,0],
                    true_zs[sample_idx,:,body_idx,1],
                    color=colors[body_idx]) 
            ax.plot(np.mean(pred_zs[:,sample_idx,:,body_idx,0], axis=0),
                    np.mean(pred_zs[:,sample_idx,:,body_idx,1], axis=0),
                    color=colors[body_idx], linestyle='--') 
            ax.scatter(pred_zs[:,sample_idx,:,body_idx,0],
                       pred_zs[:,sample_idx,:,body_idx,1],
                       color=colors[body_idx], alpha=0.005) 

                # xx, yy, f = kde(pred_zs[:,sample_idx,t,body_idx,0],
                #                 pred_zs[:,sample_idx,t,body_idx,1])

                # ax.contourf(xx, yy, f, color=colors[body_idx])
            # ax.scatter(np.mean(pred_zs[:,sample_idx,10,body_idx,0]), 
            #            np.mean(pred_zs[:,sample_idx,10,body_idx,1]), 
            #            color="black")#colors[body_idx])

        # def animate(t):
        #     for body_idx, d in enumerate(body_densities):
        #         xx, yy, f = kde(pred_zs[:,sample_idx,t,body_idx,0],
        #                         pred_zs[:,sample_idx,t,body_idx,1])
        #         ax.contourf(xx, yy, f, color=colors[body_idx])

        # fig = plt.figure()
        # print(list(range(1,num_ts)))
        # ani = animation.FuncAnimation(fig, animate, frames=range(1,num_ts), repeat=True)
        plt.show()
        plt.close()

        # for t in range(num_ts):
        #     for body_idx in range(num_bodies):
        #         viz_static_body(pred_zs[:,sample_idx,t,body_idx,:],
        #                         true_zs[sample_idx,t,body_idx,:],
        #                         body_idx+1, colors[body_idx])


        break
