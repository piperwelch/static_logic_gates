import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def Force(N, x, y, D, Lx, Ly, k_list):
    x_cont = []
    y_cont = []
    F_cont = []

    for n in range(N - 1):
        for m in range(n+1, N):
            dy = y[m] - y[n]
            Dmn = 0.5 * (D[m] + D[n])
            if abs(dy) < Dmn:
                dx = x[m] - x[n]
                if abs(dx) < Dmn:
                    dmn = np.sqrt(dx**2 + dy**2)
                    if dmn < Dmn:
                        x_cont.append([x[m], x[n]])
                        y_cont.append([y[m], y[n]])
                        F_cont.append(k_list[n] * k_list[m] / (k_list[n] + k_list[m]) * (Dmn - dmn))

    # particle wall force
    for n in range(N):
        xp = x[n]
        yp = y[n]
        r = 0.5 * D[n]

        if xp < r:
            x_cont.append([xp, 0.0])
            y_cont.append([yp, yp])
            F_cont.append(k_list[n] * (r - xp))
        elif xp > Lx - r:
            x_cont.append([xp, Lx])
            y_cont.append([yp, yp])
            F_cont.append(k_list[n] * (xp - Lx + r))
        if yp < r:
            x_cont.append([xp, xp])
            y_cont.append([yp, 0])
            F_cont.append(k_list[n] * (r - yp))
        elif yp > Ly - r:
            x_cont.append([xp, xp])
            y_cont.append([yp, Ly])
            F_cont.append(k_list[n] * (yp - Ly + r))

    return x_cont, y_cont, F_cont

def ConfigPlot_DiffSize(N, x, y, D, Lx, Ly, k_list, cn_on = 1, mark_print = 0, fn = ''):

    Dmin = np.mean(D)
    fig, ax = plt.subplots(subplot_kw = {'aspect': 'equal'})
    norm = mcolors.Normalize(vmin=0.5, vmax=10)
    cmap = cm.viridis
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # plot disks     
    linewidths=list((np.array(D) - 1) *200)
    print(linewidths)
    for i in range(N):
        edge_color = cmap(norm(k_list[i]))
        # print(edge_color)
        # alpha = 0.7 if D[i] > 1.0 else 0.3
        print(float(linewidths[i]))
        ax.add_patch(plt.Circle((x[i], y[i]), 0.5 * D[i], 
                                facecolor=edge_color, edgecolor='black', linewidth=float(linewidths[i])))

    ax.add_patch(plt.Rectangle([0.0, 0.0], Lx, Ly, \
                facecolor = 'none', edgecolor = 'k', linewidth = 1))
    # plot contact network, linewidth proportional to force magnitude
    if cn_on == 1:
        x_cont, y_cont, F_cont = Force(N, x, y, D, Lx, Ly, k_list)
        Fmin = min(F_cont)
        F_span = max(F_cont) - Fmin
        for i in range(len(x_cont)):
            ax.plot(x_cont[i], y_cont[i], color = 'k',  linewidth = 1.5 + (F_cont[i] - Fmin) / F_span * 1.0)
                
    ax.set_xlim(-0.1, Lx + 0.1)
    ax.set_ylim(-0.1, Ly + 0.1)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(0)
    plt.tight_layout()
    # save figure
    if (mark_print == 1) and len(fn) > 0:
        fig.savefig(fn, dpi = 300)
