import pandas as pa
import glob
import matplotlib.pyplot as plt

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times New Roman']

dims = [2,4,8,16,32,147]
filters = ['0.0', '0.3', '0.6', '0.9']

class_labels = ['class_acc', 'neutr_acc','contr_acc', 'ent_acc']
other_feat = ['adverse_acc', 'prem_dist', 'avg_loss']

dim_t_types= ['acc_test', 'avg_loss', 'aug_dev_acc']

mt = pa.read_csv('results\mt_metrics.csv')
mt = mt.set_index(['Model'])
mt_cols = ['8-150-' + str(d) for d in dims]
mt = mt.loc[mt_cols]
mt.index = dims
mt.index.name = 'Latent Dimension'

dataset_eval={}
for t in dim_t_types:
    dataset_eval[t] = pa.DataFrame(index = dims, columns = filters)
    dataset_eval[t].index.name = 'Latent Dimension'

class_eval = pa.DataFrame(index = dims, columns = class_labels)
class_eval.index.name = 'Latent Dimension'

other_eval = pa.DataFrame(index = dims, columns = other_feat)
other_eval.index.name = 'Latent Dimension'



train_losses = pa.Series(index = dims)
for d in dims:
    eval_table = pa.read_csv('models/real8-150-' + str(d) + '/total_eval.csv')
    train_losses[d] = pa.read_csv('models/real8-150-'+ str(d) + '/history.csv')['loss'].iloc[-1]

    for f in filters:
        for t in dim_t_types:
            dataset_eval[t][f][d] = eval_table[eval_table['threshold'] == f][t].iloc[0]
    
    class_eval.loc[d] = eval_table[eval_table['threshold'] == '0.0'][class_labels].iloc[0]
    other_eval.loc[d] = eval_table[eval_table['threshold'] == '0.0'][other_feat].iloc[0]      


eval_compare = pa.concat([other_eval, mt, dataset_eval['acc_test'][['0.0']]], axis = 1)
eval_compare['adverse_acc'] = 1 - eval_compare['adverse_acc']
eval_compare['avg_loss'] *= -1
shuffle = ['prem_dist', 'rouge', 'avg_loss', 'meteor', 'adverse_acc', '0.0']
eval_compare= eval_compare[shuffle]
comp = eval_compare.plot(logx = True, subplots = True, figsize=(8,8), layout=(3,2), legend=False, 
                         style = 'D-', grid = True, xticks = eval_compare.index)

for c,t in zip (comp.ravel(), ['Premise-Hypothesis Distance', 'ROUGE-L', 'Log-likelihood', 'METEOR', 
                               'Discriminator Error Rate', 'Classifier Accuracy']):
    c.set_xticklabels(eval_compare.index, rotation = 0)
    c.set_title(t)
comp.ravel()[-1].get_lines()[0].set_color('r')
plt.tight_layout()
plt.savefig('results/other_eval.pdf')
              

dataset_eval['avg_loss']['train_loss'] = train_losses


#orig jaccard = 0.7424
dataset_graph = {}
for t in dim_t_types:
    dataset_graph[t]= dataset_eval[t].plot(logx=True, xticks=dataset_eval[t].index, grid = True, style='D-')
    dataset_graph[t].set_xticklabels(dataset_eval[t].index)


#dataset_graph['acc_test'].set_title('Accuracies of classifiers on original test set')
dataset_graph['acc_test'].get_figure().savefig('results/gen_class_acc.pdf')

#dataset_graph['aug_dev_acc'].set_title('Accuracies of classifiers on generated dev. set')
dataset_graph['aug_dev_acc'].get_figure().savefig('results/gen_class_dev_acc.pdf')

class_eval.columns = ['All', 'Neutral', 'Contradiction', 'Entailment']
class_graph= class_eval.plot(title = 'Accuracies of datasets', logx=True, xticks=class_eval.index, 
                             grid = True, style=['D-','-','-','-'])
class_graph.set_xticklabels(class_eval.index)
class_graph.get_figure().savefig('results/data_acc.pdf')

#other_eval['adverse_acc'] = 1 - other_eval['adverse_acc']
other_eval.columns = ['Adversarial Accuracy', 'Avg. Premise-Hypothesis Distance', 'NLL']

other_graph= other_eval.plot(logx=True, xticks=other_eval.index, grid = True, style='D-')
other_graph.set_xticklabels(other_eval.index)
#other_graph.get_figure().savefig('results/other_eval_old.pdf')
       
       
       
#In [6]: cmodel.evaluate([dev[0], dev[1]], dev[2])
#8902/8902 [==============================] - 11s
#Out[6]: [0.45423631975019252, 0.82296113234320478]

#In [7]: cmodel.evaluate([test[0], test[1]], test[2])
#8830/8830 [==============================] - 10s
#Out[7]: [0.47529191597536763, 0.81268403173708026]

models = [fn.split('\\')[1] for fn in glob.glob('models\\real*-8\\total_eval.csv')]
alter_filters = ['0.0', '0.6']
other_columns = ['avg_loss', 'aug_dev_acc', 'adverse_acc', 'hypo_dist', 'total_params', 'atrain_params', 'class_acc']
other_data = pa.DataFrame(index = models, columns = other_columns)
all_models = pa.DataFrame(index = filters, columns = models)
for fn in models:
    table = pa.read_csv('models\\' + fn + '\\total_eval.csv')
    table['threshold'] = table['threshold'].astype(str)
    for f in filters:
        if len(table[table['threshold'] == f]) > 0:
            all_models[fn][f] = table[table['threshold'] == f]['acc_test'].iloc[0]
    if len(table[table['threshold'] == '0.0']) > 0:
        other_data.loc[fn] = table[table['threshold'] == '0.0'][other_columns].iloc[0]

ac = ['GenDecoder-NoClassInput', 'EndocerDecoder', 'VAE-EncoderDecoder', 'Attention-GenDecoder', 'GenDecoder']
all_models.columns = ac 
shuffle_col = ac[1:3] + [ac[4], ac[3], ac[0]] 
all_models = all_models[shuffle_col]
all_models.index.name = 'Threshold'
all_models.columns.name = 'Models'
allm_graph = all_models.plot(kind='bar', figsize=(9,9), ylim=(0.4, 0.8))
#allm_graph.get_figure().savefig('results/all_models.pdf')

#od_graph = other_data.transpose().plot(kind='bar', figsize=(9,9))

        
    


