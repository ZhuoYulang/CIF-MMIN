# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def IF_MMIN():
    WA = [0.5625,0.5333,0.6804,0.6569,0.7427,0.7317,0.6513]
    UA = [0.5844,0.5134,0.6926,0.6685,0.7565,0.7398,0.6592]
    return WA, UA


def V2():
    WA = [0.5588,0.5420,0.6768,0.6529,0.7388,0.7296,0.6498]
    UA = [0.5777,0.5191,0.6906,0.6586,0.7537,0.7374,0.6562]
    return WA, UA

def V3():
    WA = [0.5602,0.5268,0.6770,0.6621,0.7394,0.7253,0.6485]
    UA = [0.5741,0.4964,0.6909,0.6681,0.7552,0.7344,0.6532]
    return WA, UA

def V4():
    WA = [0.5512,0.5216,0.6790,0.6407,0.7415,0.7256,0.6433]
    UA = [0.5702,0.5043,0.6912,0.6525,0.7535,0.7363,0.6513]
    return WA, UA

def V5():
    WA = [0.5514,0.5259,0.6727,0.6432,0.7403,0.7293,0.6438]
    UA = [0.5759,0.5108,0.6836,0.6600,0.7528,0.7369,0.6534]
    return WA, UA

def V6():
    WA = [0.5586,0.5147,0.6728,0.6526,0.7470,0.7299,0.6459]
    UA = [0.5795,0.5026,0.6855,0.6584,0.7588,0.7381,0.6538]
    return WA, UA

def Draw_Figure(version, WA_list, UA_list, markers, save_Type='WA'):
    x = ['{a}', '{v}', '{t}', '{a, v}', '{a, t}', '{v, t}', 'Average']
    IF_MMIN_WA, IF_MMIN_UA = IF_MMIN()
    IF_MMIN_WA, IF_MMIN_UA = hundred(IF_MMIN_WA), hundred(IF_MMIN_UA)


    plt.figure(dpi=600)

    plt.grid(zorder=0, linewidth="0.5", linestyle="-.")
    plt.xlabel('Testing Conditions', fontsize=15)  # 设置x轴的标签
    # plt.title(f'{version}')

    plt.ylim(48, 77)
    plt.yticks(np.arange(48, 77, 5), fontsize=15)
    plt.xticks(fontsize=13)

    save_Type = 'WA'
    plt.ylabel('Weighted Accuracy (%)', fontsize=15)  # 设置y轴的标签
    plt.legend(loc='best')
    plt.plot(x, IF_MMIN_WA, linewidth=0.5, color="blue", marker="s", label='CIF-MMIN', linestyle='-.')
    for i in range(len(IF_MMIN_WA)):
        plt.text(x[i], IF_MMIN_WA[i] + 1, '%s'%round(IF_MMIN_WA[i], 3), ha='center', fontsize=12)

    for i, WA in enumerate(WA_list):
        plt.plot(x, WA, linewidth=0.5, color="red", marker=markers[i], label=f'{version[i]}', linestyle='-')
        for j in range(len(WA)):
            plt.text(x[j], WA[j] - 2, '%s'%round(WA[j], 3), ha='center', fontsize=12, va='bottom')
    plt.savefig(f'{save_Type}.jpg')
    plt.clf()

    plt.figure(dpi=600)

    plt.grid(zorder=0, linewidth="0.5", linestyle="-.")
    plt.xlabel('Testing Conditions', fontsize=15)  # 设置x轴的标签

    plt.ylim(48, 77)
    plt.yticks(np.arange(48, 77, 5), fontsize=15)
    plt.xticks(fontsize=13)

    save_Type = 'UA'
    plt.ylabel('Unweighted Accuracy (%)', fontsize=15)  # 设置y轴的标签
    plt.legend(loc='best')
    plt.plot(x, IF_MMIN_UA, linewidth=0.5, color="orange", marker="o", label='CIF-MMIN', linestyle='-.')
    for i in range(len(IF_MMIN_UA)):
        plt.text(x[i], IF_MMIN_UA[i] + 1, '%s'%round(IF_MMIN_UA[i], 3), ha='center', fontsize=12)
    # if version != 'V5':
    #     plt.plot(x, UA, linewidth=0.5, color="green", marker=UA_marker, label=f'{version}', linestyle='dotted')
    # else:
    #     plt.plot(x, UA, linewidth=0.5, color="green", marker=UA_marker, label=f'{version}', linestyle='dotted', markerfacecolor='white')
    for i, UA in enumerate(UA_list):
        plt.plot(x, UA, linewidth=0.5, color="green", marker=markers[i], label=f'{version[i]}', linestyle='dotted')
        for j in range(len(UA)):
            plt.text(x[j], UA[j] - 2, '%s'%round(UA[j], 3), ha='center', fontsize=12, va='bottom')
    plt.savefig(f'{save_Type}.jpg')
    plt.clf()

def hundred(list):
    return [i * 100 for i in list]


if __name__ == '__main__':
    V1_WA, V1_UA = V3()
    V2_WA, V2_UA = V4()
    V3_WA, V3_UA = V2()
    V4_WA, V4_UA = V5()
    V5_WA, V5_UA = V6()

    version = ['V1', 'V2','V3','V4','V5']
    WA_list = [hundred(V1_WA), hundred(V2_WA),hundred(V3_WA),hundred(V4_WA),hundred(V5_WA)]
    UA_list = [hundred(V1_UA), hundred(V2_UA), hundred(V3_UA), hundred(V4_UA), hundred(V5_UA)]
    marks = ['^', 'v','*','x','o']

    Draw_Figure(version, WA_list, UA_list, marks)
    # save_type = 'WA'
    # Draw_Figure('V1', hundred(V1_WA), hundred(V1_UA), '^', '^', save_Type=save_type)
    # Draw_Figure('V2', hundred(V2_WA), hundred(V2_UA), 'v', 'v', save_Type=save_type)
    # Draw_Figure('V3', hundred(V3_WA), hundred(V3_UA), '*', '*', save_Type=save_type)
    # Draw_Figure('V4', hundred(V4_WA), hundred(V4_UA), 'x', 'x', save_Type=save_type)
    # Draw_Figure('V5', hundred(V5_WA), hundred(V5_UA), 'o', 'o', save_Type=save_type)
    # save_type = 'UA'
    # Draw_Figure('V1', hundred(V1_WA), hundred(V1_UA), '^', '^', save_Type=save_type)
    # Draw_Figure('V2', hundred(V2_WA), hundred(V2_UA), 'v', 'v', save_Type=save_type)
    # Draw_Figure('V3', hundred(V3_WA), hundred(V3_UA), '*', '*', save_Type=save_type)
    # Draw_Figure('V4', hundred(V4_WA), hundred(V4_UA), 'x', 'x', save_Type=save_type)
    # Draw_Figure('V5', hundred(V5_WA), hundred(V5_UA), 'o', 'o', save_Type=save_type)