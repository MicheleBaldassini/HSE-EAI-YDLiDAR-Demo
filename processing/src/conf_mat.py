# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import transforms
# from matplotlib.collections import QuadMesh


def plot_confusion_matrix(ax, matrix, labels, fontsize=8):

    labels.append('')

    ax.set_xticks([x for x in range(len(labels))])
    ax.set_yticks([y for y in range(len(labels))])

    # Place labels on minor ticks
    ax.set_xticks([x + 0.5 for x in range(len(labels))], minor=True)
    ax.set_xticklabels(labels, rotation=30, fontsize=fontsize, minor=True)
    ax.set_yticks([y + 0.5 for y in range(len(labels))], minor=True)
    ax.set_yticklabels(labels[::-1], rotation=30, fontsize=fontsize, minor=True)

    # Hide major tick labels
    ax.tick_params(which='major', labelbottom='off', labelleft='off')

    # Finally, hide minor tick marks
    ax.tick_params(which='minor', width=0)
    
    sum_row = np.sum(matrix, axis=1)
    sum_col = np.sum(matrix, axis=0)
    diagonal = np.diagonal(matrix)

    # print('sum_row', sum_row)
    # print('sum_col', sum_col)
    # print('diagonal', diagonal)

    # Plot heat map
    # proportions = [1. * row / sum(row) for row in matrix]
    proportions = (matrix / np.sum(matrix)).tolist()

    for i in range(len(proportions)):
        proportions[i] = np.append(proportions[i], diagonal[i] / sum_row[i])  

    proportions.append(np.divide(diagonal, sum_col))
    proportions[-1] = np.append(proportions[-1], np.trace(matrix) / np.sum(matrix))

    new_matrix = np.zeros((matrix.shape[0] + 1, matrix.shape[1] + 1))
    new_matrix[:-1, :-1] = matrix.copy()

    new_matrix[:-1, -1] = sum_row
    new_matrix[-1, :-1] = sum_col
    new_matrix[-1, -1] = np.trace(matrix)

    color = np.zeros((matrix.shape[0] + 1, matrix.shape[1] + 1))
    color = np.dstack([color] * 4)
    
    for i in range(color.shape[0]):
        for j in range(color.shape[1]):
            if i == matrix.shape[0] - j and i != 0 and j != matrix.shape[1]:
                color[i, j, 0] = 188 / 255
                color[i, j, 1] = 230 / 255
                color[i, j, 2] = 196 / 255
                color[i, j, 3] = 255 / 255
            elif i == 0 and j == matrix.shape[1]:
                color[i, j, 0] = 217 / 255
                color[i, j, 1] = 217 / 255
                color[i, j, 2] = 217 / 255
                color[i, j, 3] = 255 / 255
            elif i == 0 or j == matrix.shape[1]:
                color[i, j, 0] = 240 / 255
                color[i, j, 1] = 240 / 255
                color[i, j, 2] = 240 / 255
                color[i, j, 3] = 255 / 255
            else:
                color[i, j, 0] = 249 / 255
                color[i, j, 1] = 196 / 255
                color[i, j, 2] = 192 / 255
                color[i, j, 3] = 255 / 255

    color = color.reshape(-1, 4)

    mesh = ax.pcolormesh(np.array(proportions[::-1]), color=color, lw=1)
    mesh.set_array(None)

    # Plot counts as text
    for i in range(len(new_matrix)):
        for j in range(len(new_matrix[i])):
            if i > 0 and j < new_matrix.shape[1] - 1:
                confusion = new_matrix[::-1][i][j]
                proportion = proportions[::-1][i][j]

                txt = '$\mathbf{%s}$\n%.1f%%' % (int(confusion), proportion * 100)

                ax.text(j + 0.5, i + 0.5, txt,
                        fontsize=fontsize,
                        horizontalalignment='center',
                        verticalalignment='center')
            else:
                proportion = proportions[::-1][i][j]
                if np.isnan(proportion):
                    proportion = 0
                lc = ['green', 'red']
                if i == 0 and j == new_matrix.shape[1] - 1:
                    ls = ['$\mathbf{%.1f\%%}$\n' % (proportion * 100) if proportion < 1 else '$\mathbf{%d\%%}$\n' % (proportion * 100),
                          '\n$\mathbf{%.1f\%%}$' % ((1.0 - proportion) * 100)]
                else:
                    ls = ['%.1f%%\n' % (proportion * 100) if proportion < 1 else '%d%%\n' % (proportion * 100),
                          '\n%.1f%%' % ((1.0 - proportion) * 100)]
                for s, c in zip(ls, lc):

                    text = ax.text(j + 0.5, i + 0.5, s, 
                                   fontsize=fontsize,
                                   horizontalalignment='center',
                                   verticalalignment='center', color=c) #, transform=ax.transData)
                    # text.draw(fig.canvas.get_renderer())
                    # ex = text.get_window_extent()
                    # t = transforms.offset_copy(text._transform, x=ex.width, units='dots')


    # Add finishing touches
    for i in range(len(new_matrix)):
        if i > 0:
            ax.axhline(i - 0.01, color='black', alpha=1, lw=1, zorder=2)
            ax.axvline(i - 0.01, color='black', alpha=1, lw=1, zorder=2)
        if i == 1:
            ax.axhline(i - 0.01, color='black', alpha=1, lw=1.5, zorder=2)
        if i == len(matrix):
            ax.axvline(i - 0.01, color='black', alpha=1, lw=1.5, zorder=2)
    # ax.grid(True, linestyle='-')

    ax.set_xlabel('$\mathbf{Output\ Class}$', fontsize=fontsize)
    ax.set_ylabel('$\mathbf{Target\ Class}$', fontsize=fontsize)

    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    ax.tick_params(axis='both', which='both', length=0)

    ax.set_box_aspect(1)
    # ax.set_title(title, fontsize=fontsize)


if __name__ == '__main__':    
    matrix = np.array([[13,  0,  1,  0,  2,  0],
                       [ 0, 50,  2,  0, 10,  0],
                       [ 0, 13, 16,  0,  0,  3],
                       [ 0,  0,  0, 13,  1,  0],
                       [ 0, 40,  0,  1, 15,  0],
                       [ 0,  0,  0,  0,  0, 20]])

    labels = ['bird', 'cat', 'deer', 'dog', 'frog', 'horse']

    fig, ax = plt.subplots(figsize=(5, 5))
    plot_confusion_matrix(ax, matrix, labels, fontsize=10)
    plt.tight_layout()
    plt.show()