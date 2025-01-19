import matplotlib.pyplot as plt
import numpy as np

features = ['SINC-AUROC', 'SINC-AUPRC', 'DREAM4-AUROC', 'DREAM4-AUPRC', 'THP1-AUROC', 'THP1-AUPRC', 'hESC-AUROC', 'hESC-AUPRC']

num_vars = len(features)
line = ["solid", "dotted", "dashed", "dashdot"]
def plot_spider(models, data, features):
    num_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  

    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    for i in range(4):
        model = models[i]
        values = data[model]
        values += values[:1]  
        ax.plot(angles, values, label=model, linewidth=2, linestyle=line[i])
        ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features, fontsize=15)

    ax.set_ylim(0, 0.8)
    ax.tick_params(axis='y', labelsize=14)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=15)

    plt.tight_layout()
    plt.savefig(f'{models[0]}.pdf', bbox_inches='tight') 
    plt.show()

all_scores = np.array([[0.63173333, 0.59293333, 0.4899    , 0.208     , 0.5261    ,
        0.3972    , 0.4997    , 0.0392    ],
       [0.7586    , 0.6838    , 0.5341    , 0.2177    , 0.5557    ,
        0.3669    , 0.5311    , 0.0376    ],
       [0.51343333, 0.53973333, 0.5417    , 0.2254    , 0.6112    ,
        0.4203    , 0.4971    , 0.0372    ],
       [0.5136    , 0.53853333, 0.5421    , 0.2231    , 0.6106    ,
        0.4205    , 0.507     , 0.0402    ],
       [0.41713333, 0.51066667, 0.5636    , 0.2286    , 0.4484    ,
        0.3546    , 0.5913    , 0.0468    ],
       [0.7768    , 0.6967    , 0.4589    , 0.1799    , 0.5506    ,
        0.3781    , 0.6008    , 0.0435    ],
       [0.40256667, 0.49883333, 0.5632    , 0.2261    , 0.4861    ,
        0.3642    , 0.5744    , 0.0462    ],
       [0.40313333, 0.4976    , 0.5741    , 0.2284    , 0.4792    ,
        0.3623    , 0.5767    , 0.0488    ],
       [0.681     , 0.58406667, 0.4908    , 0.1919    , 0.6261    ,
        0.3852    , 0.4198    , 0.0261    ],
       [0.76056667, 0.6372    , 0.4995    , 0.2034    , 0.5251    ,
        0.3412    , 0.4871    , 0.0294    ],
       [0.53756667, 0.5373    , 0.4999    , 0.1856    , 0.5956    ,
        0.39      , 0.1955    , 0.0199    ],
       [0.54153333, 0.53883333, 0.504     , 0.1846    , 0.6067    ,
        0.3798    , 0.1842    , 0.0196    ],
       [0.50566667, 0.52856667, 0.4806    , 0.1705    , 0.5338    ,
        0.3486    , 0.5971    , 0.0534    ],
       [0.55416667, 0.54836667, 0.5712    , 0.2452    , 0.4808    ,
        0.3302    , 0.6233    , 0.0641    ],
       [0.50323333, 0.52306667, 0.4791    , 0.1772    , 0.5521    ,
        0.3482    , 0.6008    , 0.0466    ],
       [0.5037    , 0.52366667, 0.4856    , 0.1666    , 0.5544    ,
        0.3498    , 0.604     , 0.0633    ]])

model_name = ['WENDY', 'TRENDY', 'nWENDY', 'bWENDY', 
              'GENIE3', 'tGENIE3','nGENIE3', 'bGENIE3', 
              'SINCERITIES', 'tSINCERITIES', 'nSINCERITIES', 'bSINCERITIES', 
              'NonlinearODEs', 'tNonlinearODEs', 'nNonlinearODEs', 'bNonlinearODEs']

for i in range(4):
    models = model_name[i*4: i*4+4]
    model_data = {}
    for j in range(i*4, i*4+4):
        model_data[model_name[j]] = list(all_scores[j])


    plot_spider(models, model_data, features)


total_score = np.sum(all_scores, axis=1)
total_score = np.round(total_score, 4)
s_n = [[total_score[i], model_name[i]] for i in range(16)]
s_n.sort(reverse=True)
values = [item[0] for item in s_n]
labels = [item[1] for item in s_n]

plt.figure(figsize=(14, 8))

bars = plt.barh(labels, values, color='skyblue')
for bar, value in zip(bars, values):
    plt.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f'{value:.4f}', va='center', fontsize=20)

plt.xlim(2.8, 3.8)
# Customize labels

plt.xlabel('', fontsize=14)
plt.ylabel('', fontsize=14)

plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.gca().invert_yaxis()  # Invert y-axis to match the order in the list
plt.savefig('all_scores.pdf', bbox_inches='tight') 
plt.show()








