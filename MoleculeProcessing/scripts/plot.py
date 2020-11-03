

import seaborn as sns
import os
import matplotlib.pyplot as plt
metric_names = [LOGP,WEIGHT,QED,SA]
sample_models = {
                 'sample'   : df_sample,
                 'train'    : df_train2,
                 'test'     : df_test2,
                 }
from scipy.stats import wasserstein_distance
for metric_name in metric_names:
    for name_model,sample_model in sample_models.items():
        d = sample_model.loc[~sample_model[VALIDITY].isnull()][metric_name]
        dist = wasserstein_distance(df_train2[metric_name],d)
        # dist = FrechetMetric()(distributions[metric_name]['MOSES'], d)
        sns.distplot(
            d, hist=False, kde=True,
            kde_kws={'shade': True, 'linewidth': 3},
            label='{0} ({1:0.2g})'.format(name_model, dist))
    plt.title(metric_name, fontsize=14)
    plt.legend()
    # plt.savefig(
    #     os.path.join(img_folder, metric_name+'.pdf')
    # )
    plt.savefig(
        os.path.join(img_folder, metric_name+'.png'),
        dpi=250
    )
    plt.close()





from matplotlib import pyplot

bins = numpy.linspace(-10, 10, 100)
x = df_sample.loc[~df_sample[LOGP].isnull()][LOGP].tolist()
y = df_test2.loc[~df_test2[LOGP].isnull()][LOGP].tolist()
pyplot.hist(x, bins, alpha=0.5, label='sampled')
pyplot.hist(y, bins, alpha=0.5, label='test dataset')
pyplot.legend(loc='upper right')
fig = sns.distplot(, hist=True, kde=True,
             bins=int(180/5), color = 'darkblue',
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 4})
