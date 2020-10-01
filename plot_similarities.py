import numpy as np
import matplotlib.pyplot as plt


def bar_plot_similarities(similarities_per_exp, legends=False):
    xs = np.linspace(1, similarities_per_exp[0].shape[0], similarities_per_exp[0].shape[0])
    width = 0.8/len(similarities_per_exp)

    for i in range(len(similarities_per_exp)):
        sim_exp = similarities_per_exp[i]
        if len(sim_exp)>0:
            print('plotting i: {}'.format(i))
            plt.bar(xs-width+i*width, np.mean(sim_exp), yerr=np.std(sim_exp), width=width)
        else:
            print('(i: {}) empty similarity measure: {}'.format(i, sim_exp))

    if legends:
        plt.legend(legends)

    plt.ylim(0., 0.7)
    plt.xticks(xs)

    plt.xlabel('Algorithm')
    plt.ylabel('Similarity')
    plt.title('Geodesic similarities per algorithm')

    plt.show()
    # plt.savefig(fname=fname)
    # plt.close()


similarities = []
for i in range(10):
    similarities.append(np.random.rand(int(10 * np.random.rand())))

bar_plot_similarities(similarities)
