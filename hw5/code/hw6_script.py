###############################################################################################
# this script is a copy of a notebook with the 6th homework for pandas and matplotlib/seaborn #
###############################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## the first part: pandas

# let's read *.ggf
def read_gff(path):
    return pd.read_csv(path, sep="\t", comment="#",
                       names=["chromosome", "source", "type", "start", "end",
                              "score", "strand", "phase", "attributes"])


gff = read_gff('../data/rrna_annotation.gff')


# let's read *.bed

def read_bed6(path):
    names = ['chromosome', 'start', 'end', 'name', 'score', 'strand']
    return pd.read_csv(path, sep='\t', names=names)


bed6 = read_bed6('../data/alignment.bed')

# reduce amount of information in the 'attributes' column
gff['attributes'] = gff['attributes'].apply(lambda x: x.split('Name=')[1].split('_')[0])

# group by types of rRNA and chromosomes
gff2 = gff.iloc[:, [0, -1]]
gff2.columns = ['Sequence', 'RNA type']
gff2 = gff2.groupby(by=['Sequence', 'RNA type']).size()

# just a more appealing way to represent the data
gff_cut = gff2.reset_index()
gff_cut = gff_cut.pivot(index='Sequence', columns='RNA type', values=0)
gff_cut = gff_cut.replace(np.nan, 0)
gff_cut.astype('int32').transpose()

# a barplot
gff2.unstack().plot.bar(figsize=(12, 5))

# a bedtools analog
new = pd.merge(gff, bed6, on="chromosome", how="outer")
new[(new['start_x'] >= new['start_y']) & (new['end_x'] <= new['end_y'])]



## the second part: matplotlib/seaborn
de = pd.read_csv('../data/diffexpr_data.tsv.gz', sep='\t')


# let's make a new column with labels
def label_maker(p, fold):
    part1 = ('Non-s', 'S')[p < 0.05]
    part2 = ('up', 'down')[fold < 0]
    return part1 + 'ignificantly ' + part2 + 'regulated'


de['labels'] = de.apply(lambda x: label_maker(x.pval_corr, x.logFC), axis=1)

# THE picture
fig, ax = plt.subplots(figsize=(11, 6))
plt.setp(ax.spines.values(), linewidth=1.5)
sns.scatterplot(data=de.sort_values('labels', key=lambda col: col.str.startswith('N')),
                x='logFC', y='log_pval', hue='labels', edgecolor=None, s=7)

plt.axvline(x=0, color='gray', linestyle='--', linewidth=1.5)
plt.axhline(-np.log10(0.05), color='gray', linestyle='--', linewidth=1.5)
plt.text(8, -np.log10(0.05), 'p value=0.05', weight='bold', fontsize=10, c='gray')

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.bf'] = 'Sans Serif:italic:bold'
plt.xlabel(r'$\mathbf{\bf{log_2(fold \ change)}}$', size=12)
plt.ylabel(r"$\mathbf{\bf{-log_{10}(p \ value \ corrected)}}$", size=12)
plt.title(r'$\mathbf{\bf{Volcano \ plot}}$', size=14)

plt.minorticks_on()
plt.xticks(weight='bold')
plt.yticks(weight='bold')
plt.xlim(de.logFC.min() - 1, de.logFC.max() + 1)
plt.ylim(de.log_pval.min() - 5, de.log_pval.max() + 5)

import matplotlib.font_manager as font_manager
import matplotlib as mpl

myfont = mpl.font_manager.FontProperties(weight='bold', family='sans-serif')
plt.legend(prop=myfont, fontsize=10, markerscale=1.2, shadow=True)

smol = de[de['labels'].str.startswith('S')].nsmallest(2, 'logFC')
chonk = de[de['labels'].str.startswith('S')].nlargest(2, 'logFC')
for i in [smol, chonk]:
    for j in range(2):
        row = i.iloc[j, :]
        plt.annotate(row['Sample'], xy=(row['logFC'], row['log_pval'] + 1),
                     xytext=(row['logFC'] - 0.5, row['log_pval'] + 12),
                     weight='bold',
                     arrowprops=dict(facecolor='r', width=3,
                                     headwidth=9, headlength=6, shrink=0.05))

plt.savefig('volcano_plot.png', dpi=200, bbox_inches='tight',
            facecolor='w', transparent=True)

# the same thing, without nice settings just other
# basics of matplotlib
label_names = list(set(de['labels']))
for i in range(4):
    lab_inter = de[de['labels'] == label_names[i]]
    plt.scatter(lab_inter['logFC'], lab_inter['log_pval'], label=label_names[i])
plt.legend()

# pandas
de.set_index('logFC').sort_index().groupby('labels')['log_pval'].plot(style='o')
plt.legend()
