import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    from utils.precision_recall import precision_recall_array


def create_pr_doughnuts(precision, recall):

    fig, ax = plt.subplots(1,2)

    colours = ['lightblue', 'blue']

    ax[0].pie(([1 - precision, precision]), startangle=90, wedgeprops=dict(width=.5), colors = colours)
    ax[0].set_title('Precision')
    ax[0].text(x = 0, y = 0, s = f'{precision*100: .1f}%', ha = 'center')

    ax[1].pie(([1 - recall, recall]), startangle=90, wedgeprops=dict(width=.5), colors = colours)
    ax[1].set_title('Recall')
    ax[1].text(x = 0, y = 0, s = f'{recall*100: .1f}%', ha = 'center')

    return fig


def create_pr_chart(pr_data, precision, recall):

    if __name__ == '__main__':
        pr_data = precision_recall_array()

    fig, ax = plt.subplots(figsize = (10,4))

    ax.plot(pr_data[:,0], pr_data[:,1])
    ax.set_xlabel('Recall (%)')
    ax.set_ylabel('Precision (%)')

    ax.scatter(recall*100, precision*100, c = 'red', marker='o', s=100)

    return fig


