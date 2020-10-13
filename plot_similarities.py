import numpy as np
import matplotlib.pyplot as plt


def bar_plot_similarities(similarities_per_lfn, xticks=False, legends=False, title=False):
    xs = np.linspace(0.2, len(similarities_per_lfn)-0.8, len(similarities_per_lfn))
    width = 0.15

    for i in range(len(similarities_per_lfn)):
        sim_exp_lfn = similarities_per_lfn[i]
        for j in range(len(sim_exp_lfn)):
            similarities_for_lfn_and_model = sim_exp_lfn[j]
            if len(similarities_for_lfn_and_model)>0:
                # print('plotting i,j: {},{}'.format(i,j))
                plt.bar(i+j*width, np.mean(similarities_for_lfn_and_model), yerr=np.std(similarities_for_lfn_and_model), width=width, color=['C0', 'C1', 'C2', 'C3'][j])
            else:
                print('(i,j: {},{}) empty similarity measure: {}'.format(i, j, similarities_for_lfn_and_model))

    if legends:
        plt.legend(legends)

    plt.ylim(0., 0.7)
    # plt.xticks(xs)
    if xticks:
        plt.xticks(xs, xticks)
    else:
        plt.xticks(xs)

    plt.xlabel('Algorithm')
    plt.ylabel('Similarity')
    if title:
        plt.title(title)
    else:
        plt.title('Geodesic similarity per experiment')

    plt.show()
    # plt.savefig(fname=fname)
    # plt.close()


# similarities = []
# for i in range(10):
#     similarities.append(np.random.rand(int(10 * np.random.rand())))
#
# bar_plot_similarities(similarities)