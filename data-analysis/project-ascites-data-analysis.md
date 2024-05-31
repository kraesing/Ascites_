```python
# Project: Ascites
# content: Data analysis

# author: kraesing
# mail: lau.kraesing.vestergaard@regionh.dk
# GitHub: https://github.com/kraesing

##########################################################################
```


```python
### IMPORT LIBRARIES

import warnings
warnings.filterwarnings("ignore")

# Libraries for file management.
import os
import glob 
import fnmatch
import shutil 

# Libraries for data management.
import pandas as pd
import numpy as np
from numpy.random import seed

# Libraries for statistical analysis.
import scipy.stats as stats
import scipy.spatial.distance import squareform, pdist

# Libraries for plotting.
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# Setting params for plotting
sns.set_theme(style="whitegrid")
sns.set_context("notebook")
plt.rcParams["font.family"] = "Helvetica"
sns.set_palette("colorblind")
sns.set_color_codes(palette="muted")

# Libraries for machine learning.
import tensorflow as tf

# Libraries for Natural Language Processing (NLP)
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Libraries for Network analysis
from pyvis.network import Network
from itertools import product
import networkx as nx

# Co-mutational plotting
from comut import comut

# Conformation
print("Libraries imported!")

##########################################################################
```


```python
### Setting seed 

np.random.seed(34)
tf.random.set_seed(34)
##########################################################################
```


```python
### Change directory

os.chdir(path + "IonTorrent5.18")

##########################################################################
```


```python
### Loading data

os.chdir(os.getcwd() + path_to_files)

samples_ascites_ = glob.glob("*600005*")
samples_tissue_ = glob.glob("*600006*")
samples_plasma_ = glob.glob("*600004*")

##########################################################################
```


```python
### Loading clinical data

df_patients_id = pd.read_excel(path_to_clinical_data, sheet_name="OvCA", dtype="str")
list_patients_id = list(set(df_patients_id["ID"]))

# Remove nan element
removed_element = list_patients_id.pop(0)

# Introduce leading zeros for ID with the length of 1.
list_patients_id = ["0"+x if len(x) < 2 else x for x in list_patients_id]

# Creating empty lists
samples_tissue = []
samples_ascites = []
samples_plasma = []

for i in samples_tissue_:
    if i[19:21] in list_patients_id:
        samples_tissue.append(i)
        
for i in samples_ascites_:
    if i[19:21] in list_patients_id:
        samples_ascites.append(i)
        
for i in samples_plasma_:
    if i[19:21] in list_patients_id:
        samples_plasma.append(i)        

##########################################################################
```


```python
### Loading Histological data

histology = pd.read_excel(r"histologi_.xlsx", sheet_name="Sheet4")
histology["ID"] = ["0"+str(x) if len(str(x)) == 1 else str(x)
                   for x in histology["ID"]]

##########################################################################
```


```python
### Counting initial variants

counting_variants(samples_ascites, samples_tissue)

##########################################################################
```


```python
### Running variant filtering for ascites samples

interval_start = 0
interval_stop = 21

mutation_list_ascites = []
mutation_rest_ascites = []
resc_ascites = []
merged_ascites = []
rescued_ascites = []
ascites_samples = []

for sample in samples_ascites:
    print("Analyzing sample: {}".format(sample[interval_start:interval_stop]))
    ascites_samples.append(sample[interval_start:interval_stop])
    df = pd.read_csv(sample, sep="\t")
    df["samplename"] = sample[interval_start:interval_stop]
    df1 = Initial_cleaning(df)
    df1 = Gene_cleaning(df1)
    df1 = Original_filtering(df=df1, allele_freq=(7), coverage=(200))
    # df1 = df1.loc[~df1["Clinical_significance"].isin(list_benigns)]
    df_pass = df1.loc[df1["true_variant"] == "PASS"]
    if len(df1) != 0:
        df2 = Variant_rescue_cleaning(df1)
        apply_fishers_test_(df=df2, strandbias_threshold=0.05)
        mutation_rest_ascites.append(df2)

    mutation_list_ascites.append(df_pass)

for i in mutation_rest_ascites:
    _ = i.copy()
    _ = _.loc[~(_["Var+"] == "0")]
    _ = _.loc[~(_["Var-"] == "0")]
    p_value = _.loc[_["true_variant"] == "Above p-value"]
    low_phred = _.loc[_["true_variant"] == "Low Phred Score"]
    base_coverage = _.loc[_["true_variant"] == "Low base coverage"]
    below_threshold = _.loc[_["true_variant"] ==
                            "Below threshold for allele frequency"]
    low_overall = _.loc[_["true_variant"] ==
                        "Low overall coverage (caution warrant)"]

    if len(low_phred) != 0:
        Variant_rescue_filtering(df=low_phred, Frequency=2.5, Pvalue=0.0001,
                                 PhredScore=40, Coverage=750, suffix="Low Phred score")

    if len(base_coverage) != 0:
        Variant_rescue_filtering(df=base_coverage, Frequency=2.5, Pvalue=0.0001,
                                 PhredScore=200, Coverage=70, suffix="Low base-coverage")

    if len(low_overall) != 0:
        Variant_rescue_filtering(df=low_overall, Frequency=2.5, Pvalue=0.0001, PhredScore=200,
                                 Coverage=70, suffix="Low base-coverage overall (caution warrant)")

    if len(p_value) != 0:
        Variant_rescue_filtering(df=p_value, Frequency=0.075, Pvalue=0.05,
                                 PhredScore=12.5, Coverage=500, suffix="Low p-value")

    if len(below_threshold) != 0:
        Variant_rescue_filtering(df=below_threshold, Frequency=2.5, Pvalue=0.0001,
                                 PhredScore=400, Coverage=1000, suffix="Below allele_threshold")

    frames = [p_value, low_phred, base_coverage, below_threshold, low_overall]
    merged_ascites_rest = pd.concat(frames)
    merged_ascites.append(merged_ascites_rest.loc[merged_ascites_rest["Rescue_filtering"].str.contains("PASS Rescue filtering")])

##########################################################################
```


```python
### Running variant filtering for tumor tissue samples

mutation_list_tissue = []
mutation_rest_tissue = []
resc_tissue = []
merged_tissue = []
rescued_tissue = []
tissue_samples = []

for sample in samples_tissue:
    print("Analyzing sample: {}".format(sample[interval_start:interval_stop]))
    tissue_samples.append(sample[interval_start:interval_stop])
    df = pd.read_csv(sample, sep="\t")
    df["samplename"] = sample[interval_start:interval_stop]
    df = remove_artifactual_variants(df)
    df = df.dropna(subset=["Ref+/Ref-/Var+/Var-"])
    df1 = Initial_cleaning(df)
    df1 = Gene_cleaning(df1)
    df1 = Original_filtering(df=df1, allele_freq=(11), coverage=(200))
    # df1 = df1.loc[~df1["Clinical_significance"].isin(list_benigns)]
    df_pass = df1.loc[df1["true_variant"] == "PASS"]
    if len(df1) != 0:
        df2 = Variant_rescue_cleaning(df1)
        apply_fishers_test_(df=df2, strandbias_threshold=0.05)
        mutation_rest_tissue.append(df2)

    mutation_list_tissue.append(df_pass)


for i in mutation_rest_tissue:
    _ = i.copy()
    _ = _.loc[~(_["Var+"] == "0")]
    _ = _.loc[~(_["Var-"] == "0")]
    p_value_t = _.loc[_["true_variant"] == "Above p-value"]
    low_phred_t = _.loc[_["true_variant"] == "Low Phred Score"]
    base_coverage_t = _.loc[_["true_variant"] == "Low base coverage"]
    below_threshold_t = _.loc[_["true_variant"]
                              == "Below threshold for allele frequency"]
    low_overall_t = _.loc[_["true_variant"] ==
                          "Low overall coverage (caution warrant)"]

    if len(low_phred_t) != 0:
        Variant_rescue_filtering(df=low_phred_t, Frequency=2.5, Pvalue=0.0001,
                                 PhredScore=40, Coverage=750, suffix="Low Phred score")

    if len(base_coverage_t) != 0:
        Variant_rescue_filtering(df=base_coverage_t, Frequency=2.5, Pvalue=0.0001,
                                 PhredScore=200, Coverage=70, suffix="Low base-coverage")

    if len(low_overall_t) != 0:
        Variant_rescue_filtering(df=low_overall_t, Frequency=2.5, Pvalue=0.0001, PhredScore=200,
                                 Coverage=70, suffix="Low base-coverage overall (caution warrant)")

    if len(p_value_t) != 0:
        Variant_rescue_filtering(df=p_value_t, Frequency=0.075, Pvalue=0.05,
                                 PhredScore=12.5, Coverage=500, suffix="Low p-value")

    if len(below_threshold_t) != 0:
        Variant_rescue_filtering(df=below_threshold_t, Frequency=2.5, Pvalue=0.0001,
                                 PhredScore=400, Coverage=1000, suffix="Below allele_threshold")

    if len(low_phred_t + base_coverage_t + p_value_t + below_threshold_t + low_overall_t) == 0:
        low_phred_t["Rescue_filtering"] = "Empty"

    frames_t = [p_value_t, low_phred_t, base_coverage_t,
                below_threshold_t, low_overall_t]
    merged_tissue_rest = pd.concat(frames_t)
    merged_tissue.append(merged_tissue_rest.loc[merged_tissue_rest["Rescue_filtering"].str.contains("PASS Rescue filtering")])

##########################################################################
```


```python
### Running variant filtering for plasma samples. NOT INCLUDED IN THE STUDY.

mutation_list_plasma = []
mutation_rest_plasma = []
resc_plasma = []
merged_plasma = []
rescued_plasma = []
plasma_samples = []

for sample in samples_plasma:
    print("Analyzing sample: {}".format(sample[interval_start:interval_stop]))
    plasma_samples.append(sample[interval_start:interval_stop])
    df = pd.read_csv(sample, sep="\t")
    df["samplename"] = sample[interval_start:interval_stop]
    df = remove_artifactual_variants(df)
    df = df.dropna(subset=["Ref+/Ref-/Var+/Var-"])
    df1 = Initial_cleaning(df)
    df1 = Gene_cleaning(df1)
    df1 = Original_filtering(df=df1, allele_freq=(7), coverage=(200))
    # df1 = df1.loc[~df1["Clinical_significance"].isin(list_benigns)]
    df_pass = df1.loc[df1["true_variant"] == "PASS"]
    if len(df1) != 0:
        df2 = Variant_rescue_cleaning(df1)
        apply_fishers_test_(df=df2, strandbias_threshold=0.05)
        mutation_rest_plasma.append(df2)

    mutation_list_plasma.append(df_pass)

for i in mutation_rest_plasma:
    _ = i.copy()
    _ = _.loc[~(_["Var+"] == "0")]
    _ = _.loc[~(_["Var-"] == "0")]
    p_value_p = _.loc[_["true_variant"] == "Above p-value"]
    low_phred_p = _.loc[_["true_variant"] == "Low Phred Score"]
    base_coverage_p = _.loc[_["true_variant"] == "Low base coverage"]
    below_threshold_p = _.loc[_["true_variant"]
                              == "Below threshold for allele frequency"]
    low_overall_p = _.loc[_["true_variant"] ==
                          "Low overall coverage (caution warrant)"]

    if len(low_phred_p) != 0:
        Variant_rescue_filtering(df=low_phred_p, Frequency=2.5, Pvalue=0.0001,
                                 PhredScore=40, Coverage=750, suffix="Low Phred score")

    if len(base_coverage_p) != 0:
        Variant_rescue_filtering(df=base_coverage_p, Frequency=2.5, Pvalue=0.0001,
                                 PhredScore=200, Coverage=70, suffix="Low base-coverage")

    if len(low_overall_p) != 0:
        Variant_rescue_filtering(df=low_overall_p, Frequency=2.5, Pvalue=0.0001, PhredScore=200,
                                 Coverage=70, suffix="Low base-coverage overall (caution warrant)")

    if len(p_value_p) != 0:
        Variant_rescue_filtering(df=p_value_p, Frequency=0.075, Pvalue=0.05,
                                 PhredScore=12.5, Coverage=500, suffix="Low p-value")

    if len(below_threshold_p) != 0:
        Variant_rescue_filtering(df=below_threshold_p, Frequency=2.5, Pvalue=0.0001,
                                 PhredScore=400, Coverage=1000, suffix="Below allele_threshold")

    if len(low_phred_p + base_coverage_p + p_value_p + below_threshold_p + low_overall_p) == 0:
        low_phred_p["Rescue_filtering"] = "Empty"

    frames_p = [p_value_p, low_phred_p, base_coverage_p,
                below_threshold_p, low_overall_p]
    merged_plasma_rest = pd.concat(frames_p)
    merged_plasma.append(merged_plasma_rest.loc[merged_plasma_rest["Rescue_filtering"].str.contains("PASS Rescue filtering")])

###############################################################################
```


```python
### Merging variants lists


t_ = pd.concat(mutation_list_tissue)
a_ = pd.concat(mutation_list_ascites)
p_ = pd.concat(mutation_list_plasma)
df_combined = pd.concat([t_, a_, p_])

t__ = pd.concat(merged_tissue)
a__ = pd.concat(merged_ascites)
p__ = pd.concat(merged_plasma)
df_combined_res = pd.concat([t__, a__, p__])

###############################################################################
```


```python
### Creating a list of columns to be included for subsetting

subset_cols = ["samplename", "Locus", "Genes1", "Genotype", "Amino Acid Change", "Raw Coverage", "Location",
               "Phred QUAL Score", "P-Value", "Allele Frequency %", "Rescue_filtering", "Variant Effect", "Clinical_significance"]

###############################################################################
```


```python
### Subsetting the data

# Creating a new column, to identify which filter the variant originates from
df_combined["Rescue_filtering"] = "Original"

# Subsetting the data with the columns defined in subset_cols
df_combined = df_combined[subset_cols]
df_combined_res = df_combined_res[subset_cols]

final_data = pd.concat([df_combined, df_combined_res])
final_data["ID_samplename"] = final_data["samplename"].apply(lambda x: x[:-3])

###############################################################################
```


```python
### Checking the number of samples

print(final_data["ID_samplename"].value_counts())

###############################################################################
```


```python
### Inspecting the number of unique genes

print(final_data.Genes1.nunique())

###############################################################################
```


```python
### Preparing data for matrix formation

# Making an ID column (mutation) for each mutation identified
final_data["mutation"] = final_data["Genes1"] + "_" + \
    final_data["Locus"] + "_" + final_data["Amino Acid Change"]
final_data["values"] = 1.0

# https://stackoverflow.com/questions/51881503/assign-a-dictionary-value-to-a-dataframe-column-based-on-dictionary-key
# Creating a dictionary to map a specific value to the dataframe
dict_biopsytype = {"600005": "Ascites",
                   "600006": "Tissue",
                   "600004": "Plasma"}

# Assigning new columns
final_data["sample"] = final_data.samplename.apply(lambda x: x[-9:-3])
final_data["samplename_alias"] = final_data["sample"].map(dict_biopsytype)
final_data["sample_ID"] = final_data["samplename"].apply(lambda x: x[-3:])
final_data["sample_ID"] = [x.replace(".", "_")
                           for x in final_data["sample_ID"]]
final_data["samplename_alias"] = final_data["samplename_alias"] + \
    final_data["sample_ID"]
final_data["biopsy_type_"] = final_data["sample"].map(dict_biopsytype)

###############################################################################
```


```python
### Filtering the data to only contain variants from ascites and tumor tissue. 

final_data = final_data.loc[(final_data["samplename_alias"].str.contains(
    "Ascites")) | (final_data["samplename_alias"].str.contains("Tissue"))]

###############################################################################
```


```python
### Identifying samples with no variants identified

# ONLY FOR ASCITES AND TUMOR
combined_sample_list = set(ascites_samples + tissue_samples)

# The unique samples found after the analysis.
identified_sample_list = set(final_data.samplename.unique())

# The following will identify samples that didn't have any variants called in the filtering.
set_diff = combined_sample_list.symmetric_difference(identified_sample_list)

print(set_diff)

###############################################################################
```


```python
### Plotting Venn Diagram for all variants identified from ascites and tumor tissue

# FOR ALL VARIANTS
# Creating a special column for a Venn diagram
final_data["for_venn"] = final_data["mutation"] + final_data["sample_ID"]

# Storing the mutations from each tissue type into a set.
ascites_venn = set(
    final_data.loc[final_data["sample"] == "600005"]["for_venn"])
tissue_venn = set(final_data.loc[final_data["sample"] == "600006"]["for_venn"])

total = len(ascites_venn.union(tissue_venn))

# Creating the Venn diagram to show overall overlap of identified variants.
plt.figure()
venn2([ascites_venn, tissue_venn],
      set_labels=("Ascites", "Tumor"),
      set_colors=("#a74e71", "#4ea784"),
      subset_label_formatter=lambda x: str(
          x) + "\n(" + f"{(x/total):1.00%}" + ")",
      alpha=1)

# Saving the plot
# save_fig(plt, "Venn_diagram_updated.svg", transparent_background=True)

###############################################################################
```


```python
### Creating list to filter for pathogenic variants

list_variants_1 = ["Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic", "Pathogeic/Likely pathogenic other", "Likely pathogenic other",
                   "Pathogenic drug response other", "Conflicting interpretations of pathogenicity CONF Likely pathogenic Uncertain significance",
                   "drug response", "Conflicting interpretations of pathogenicity CONF Pathogenic Likely pathogenic Uncertain significance",
                   "Conflicting interpretations of pathogenicity CONF Pathogenic Uncertain significance", "Pathogenic/Likely pathogenic other", "Likely pathogenic*"]

###############################################################################
```


```python
### Filtering the data frame 

final_data_pat = final_data.loc[final_data["Clinical_significance"].isin(
    list_variants_1)]

###############################################################################
```


```python
### Identifying samples with no of pathogenic and likely pathogenic variants

# ONLY FOR ASCITES AND TUMOR
combined_sample_list = set(ascites_samples + tissue_samples)
# The unique samples found after the analysis. 
identified_sample_list_pat = set(final_data_pat.samplename.unique())

# The following will identify samples that didn't have any pathogenic variants in the filtering.
set_diff_ = combined_sample_list.symmetric_difference(identified_sample_list_pat)

###############################################################################
```


```python
### Plotting Venn Diagram for pathogenic variants identified from ascites and tumor tissue

# Storing the mutations from each tissue type into a set.
ascites_venn_pat = set(
    final_data_pat.loc[final_data_pat["sample"] == "600005"]["for_venn"])
tissue_venn_pat = set(
    final_data_pat.loc[final_data_pat["sample"] == "600006"]["for_venn"])

total_pat = len(ascites_venn_pat.union(tissue_venn_pat))
# Creating a venn diagram to show overall overlap of identified mutations.
plt.figure()
vd_pat = venn2([ascites_venn_pat, tissue_venn_pat],
               set_labels=("Ascites", "Tumor"),
               set_colors=("#a74e71", "#4ea784"),
               subset_label_formatter=lambda x: str(
                   x) + "\n(" + f"{(x/total_pat):1.00%}" + ")",
               alpha=1)

# Saving the plot
# save_fig(plt, "Venn_diagram_pathogenic_updated.svg", transparent_background=True)

###############################################################################
```


```python
### Calculating Pearson correlation for all variants and pathogenic variants, respectively. 

# all variants
r, p, df_pearson = pearson_coeff(final_data, "Ascites", "Tissue")

# pathogenic variants
r_pat, p_pat, df_pearson_pat = pearson_coeff(final_data_pat, "Ascites", "Tissue")

###############################################################################
```


```python
### Plotting box plot for allele frequency comparison of pathogenic variants between ascites and tumor tissue 

box_plot_data = final_data_pat[["Location", "Allele Frequency %", "biopsy_type_", "for_venn"]]

list_diff_box = set(box_plot_data.loc[box_plot_data["biopsy_type_"] == "Ascites"]["for_venn"]).symmetric_difference(set(box_plot_data.loc[box_plot_data["biopsy_type_"] == "Tissue"]["for_venn"]))

final_data_pat_box = final_data_pat.loc[final_data_pat["for_venn"].isin(list_diff_box)]

palette = {"Ascites": "#a74e71",
           "Tissue": "#4ea784",
           "Plasma": "#a7844e"}


plt.figure(figsize=(2, 10))
sns.boxplot(x="biopsy_type_", y="Allele Frequency %",
            data=final_data_pat, palette=palette)
sns.swarmplot(x="biopsy_type_", y="Allele Frequency %", data=final_data_pat,
              palette=palette, edgecolor="black", linewidth=1, marker="s")
sns.swarmplot(x="biopsy_type_", y="Allele Frequency %", data=final_data_pat_box,
              color="black", edgecolor="black", linewidth=1, marker="s")


plt.ylabel("Allele frequency (%)")
plt.xlabel("")

# Saving the plot
# save_fig(plt, "Boxplot_swarmplot_pat_updated.svg", False)

###############################################################################
```


```python
### Identifying data distribution for parametric or non-parametric test

# If the p-value is less than 0.05, the null hypothesis is rejected of the shapiro-Wilk test.
# Meaning that the samples does not come from a normal distribution.

# SHAPIRO-WILK
print(stats.shapiro(
    final_data_pat.loc[final_data_pat["biopsy_type_"] == "Ascites"]["Allele Frequency %"]))

print(stats.shapiro(
    final_data_pat.loc[final_data_pat["biopsy_type_"] == "Tissue"]["Allele Frequency %"]))

# Mann-Whitney U-Test
# https://www.reneshbedre.com/blog/mann-whitney-u-test.html?utm_content=cmp-true

s_MW, p_MW = stats.mannwhitneyu(final_data_pat.loc[final_data_pat["biopsy_type_"] == "Ascites"]["Allele Frequency %"],
                                final_data_pat.loc[final_data_pat["biopsy_type_"]
                                                   == "Tissue"]["Allele Frequency %"],
                                alternative='two-sided')

###############################################################################
```


```python
### Plotting Pearson correlation for all identified variants

# All variants
sns.set_theme(style="whitegrid")
sns.lmplot(x="AF_Tissue",
           y="AF_Ascites",
           data=df_pearson,
           markers="s",
           line_kws=({"color": "#ffa012"}),  
           scatter_kws=({"color": "#5c5954"}))

sns.scatterplot(x="AF_Tissue", y="AF_Ascites", data=df_pearson,
                marker="s", edgecolor="black", color="#be6563")
plt.xlabel("Tumor -\n Allele frequency (%)", fontsize=12)
plt.ylabel("Ascites -\n Allele frequency (%)", fontsize=12)
plt.xlim(-2.5, 102.5)
plt.ylim(-2.5, 102.5)
plt.legend(["pearsonr = %s \n p = %f" % (r, p)],
           fontsize=12, bbox_to_anchor=(1.5, 0.99))

sns.scatterplot(x="AF_Tissue", y="AF_Ascites", data=df_pearson_pat,
                marker="s", edgecolor="black", color="white")
plt.xlabel("Tumor Tissue -\n Allele frequency (%)", fontsize=12)
plt.ylabel("Ascites -\n Allele frequency (%)", fontsize=12)
plt.xlim(-2.5, 102.5)
plt.ylim(-2.5, 102.5)

# Saving the plot
# save_fig(plt, "correlation_updated.svg", False)

###############################################################################
```


```python
### Plotting Pearson correlation for pathogenic variants

sns.set_theme(style="whitegrid")
sns.lmplot(x="AF_Tissue",
           y="AF_Ascites",
           data=df_pearson_pat,
           markers="s",
           line_kws=({"color": "#ffa012"}),
           scatter_kws=({"color": "#5c5954"}))

sns.scatterplot(x="AF_Tissue", y="AF_Ascites", data=df_pearson_pat,
                marker="s", edgecolor="black", color="white")
plt.xlabel("Tumor Tissue -\n Allele frequency (%)", fontsize=12)
plt.ylabel("Ascites -\n Allele frequency (%)", fontsize=12)
plt.xlim(-2.5, 102.5)
plt.ylim(-2.5, 102.5)
plt.legend(["pearsonr = %s \n p = %f" % (r_pat, p_pat)],
           fontsize=12, bbox_to_anchor=(1.5, 0.99))

# Saving the plot
# save_fig(plt, "correlation_pathogenic_updated.svg", False)

###############################################################################
```


```python
### Creating matrix for all variants from ascites and tumor tissue

# Creating pivot table for all variants
pivot = pd.pivot_table(
    columns="samplename_alias", index="mutation", values="values", data=final_data)

# Creating the matrix for samples in absence of variants
print(set_diff)
set_diff_df = pd.DataFrame(index=set_diff, columns=pivot.columns)
set_diff_df["samplename"] = list(set_diff)
set_diff_df.index = [x[-9:-3] for x in set_diff_df.index]
set_diff_df.index = set_diff_df.index.map(dict_biopsytype)
set_diff_df["samplename"] = set_diff_df["samplename"].apply(lambda x: x[-3:])
set_diff_df["samplename"] = [x.replace(".", "_")
                             for x in set_diff_df["samplename"]]
set_diff_df.index = set_diff_df.index + set_diff_df["samplename"]
set_diff_df.drop(columns=["samplename"], inplace=True)

# Merging the set_diff with the matrix for samples with variants.
pivot = pd.concat([pivot, set_diff_df])

# Zero padding the NaN values.
pivot.fillna(0, inplace=True)

pivot = pivot.rename_axis(None, axis=1)

order_ascites = list(pivot.columns.unique()[0:32])
order_tissue = list(pivot.columns.unique()[32:])
ordered_ = []

for i,k in zip(order_ascites, order_tissue):
    ordered_.append(i)
    ordered_.append(k)

pivot = pivot[ordered_]
pivot.columns = [col.replace("_", " ") for col in pivot.columns]
pivot.columns = [col.replace("Tissue ", "TT") for col in pivot.columns]
pivot.columns = [col.replace("Ascites ", "A") for col in pivot.columns]

###############################################################################
```


```python
### Creating matrix for pathogenic variants identified from ascites and tumor tissue

# Creating pivot table for pathogenic variants 
pivot_pat = pd.pivot_table(
    columns="samplename_alias", index="mutation", values="values", data=final_data_pat)

 # Zeropadding the NaN values.
pivot_pat.fillna(0, inplace=True)

# Creating the matrix for samples with the absent of variants
print(set_diff_)
set_diff_df_pat = pd.DataFrame(index=set_diff_, columns=pivot_pat.index)
set_diff_df_pat["samplename"] = list(set_diff_)
set_diff_df_pat.index = [x[-9:-3] for x in set_diff_df_pat.index]
set_diff_df_pat.index = set_diff_df_pat.index.map(dict_biopsytype)
set_diff_df_pat["samplename"] = set_diff_df_pat["samplename"].apply(lambda x: x[-3:])
set_diff_df_pat["samplename"] = [x.replace(".", "_") for x in set_diff_df_pat["samplename"]]
set_diff_df_pat.index = set_diff_df_pat.index + set_diff_df_pat["samplename"]
set_diff_df_pat.drop(columns=["samplename"], inplace=True)
set_diff_df_pat["wildtype"] = 0.1
set_diff_df_pat = set_diff_df_pat.T

# Merging the set_diff with the matrix for samples with variants.
pivot_pat = pd.concat([pivot_pat, set_diff_df_pat], axis=1)
pivot_pat = pivot_pat[ordered_]

# Zero padding the NaN values.
pivot_pat.fillna(0, inplace=True)

pivot_pat = pivot_pat.rename_axis(None, axis=1)


pivot_pat.columns = [col.replace("_", " ") for col in pivot_pat.columns]
pivot_pat.columns = [col.replace("Tissue ", "TT") for col in pivot_pat.columns]
pivot_pat.columns = [col.replace("Ascites ", "A") for col in pivot_pat.columns]

###############################################################################
```


```python
### Computing correlation matrix

# For all variants
corr_mat = pivot.corr().stack().reset_index(name="correlation")
corr_mat["size"] = np.nan
corr_mat["size"] = [0 if x <= 0 else x for x in corr_mat["correlation"]]
corr_mat["size"] = [0.2 if x > 0 and x <= 0.3 else x for x in corr_mat["size"]]
corr_mat["size"] = [0.4 if x > 0.3 and x <= 0.5 else x for x in corr_mat["size"]]
corr_mat["size"] = [0.6 if x > 0.5 and x <= 0.7 else x for x in corr_mat["size"]]
corr_mat["size"] = [0.8 if x > 0.7 and x < 0.9 else x for x in corr_mat["size"]]
corr_mat["size"] = [1 if x > 0.9 else x for x in corr_mat["size"]]


# For pathogenic variants
corr_mat_pat = pivot_pat.corr().stack().reset_index(name="correlation")
corr_mat_pat["size"] = np.nan
corr_mat_pat["size"] = [0 if x <= 0 else x for x in corr_mat_pat["correlation"]]
corr_mat_pat["size"] = [0.2 if x > 0 and x <= 0.3 else x for x in corr_mat_pat["size"]]
corr_mat_pat["size"] = [0.4 if x > 0.3 and x <= 0.5 else x for x in corr_mat_pat["size"]]
corr_mat_pat["size"] = [0.6 if x > 0.5 and x <= 0.7 else x for x in corr_mat_pat["size"]]
corr_mat_pat["size"] = [0.8 if x > 0.7 and x < 0.9 else x for x in corr_mat_pat["size"]]
corr_mat_pat["size"] = [1 if x > 0.9 else x for x in corr_mat_pat["size"]]

###############################################################################
```


```python
### Plotting correlation

scatter_heatplot(data=corr_mat, x= "level_0", y="level_1", hue="correlation", title="Corr_plot_vlag_.svg")

scatter_heatplot(data=corr_mat_pat, x= "level_0", y="level_1", hue="correlation", title="Corr_plot_pat_vlag_.svg")

###############################################################################
```


```python
### Preparing for NLP of all variants

# Preparing data for NLP analysis
df_nlp = prep_nlp(final_data, set_diff)

corpus = []

for i in df_nlp["Mutation"]:
    corpus.append(i)

# Saving variant files for samples separately 
#saving_mut_files(df_nlp, "_all_updated")

###############################################################################
```


```python
### Creating object for NLP for all variants

# Counting the number of variants identified in each sample.
df_nlp["Mutation"].apply(lambda x: len(str(x).split(", "))).sum()

# Calculating Cosine similarities.
co_sim1_df, doc1_df = Cosine_similarity(corpus=corpus)

# Getting the ID of the samples as index and column names. 
id_names = df_nlp.Samplename
id_names = id_names.rename("id_names")
col_names = df_nlp.Samplename
col_names = col_names.rename("col_names")

# Renameing the index and the columns.
co_sim1_df.index = id_names
co_sim1_df.columns = col_names

co_sim1_df.columns = [col.replace("T", "TT") for col in co_sim1_df.columns]
co_sim1_df.index = [idx.replace("T", "TT") for idx in co_sim1_df.index]

# Plotting the clustermap
plt.figure()
sns.clustermap(co_sim1_df, cmap="Reds", yticklabels=True, xticklabels=True, annot=False, figsize=(12,12), dendrogram_ratio=0.15)

# Saving the plot
# save_fig(plt, "clustermap_all.svg", False)

###############################################################################
```


```python
### Preparing for network analysis
  
# applying the product method
print(list(product(co_sim1_df.index, co_sim1_df.columns)))

# Creating series for input for the network analysis
sources, targets, weights = source_targets_weights(co_sim1_df)

###############################################################################
```


```python
### Creating and plotting the network with cosine similarity as weights for all variants. 

import random
import numpy as np

# Seed must be run in the same cell for reproducibility
seed = 124
random.seed(seed)
np.random.seed(seed)

# Plotting the network
plot_network(sources = sources, targets = targets, weights = weights)

# Saving the plot
#save_fig(plt, "network_updated.svg", True)

###############################################################################
```


```python
### Preparing for NLP of pathogenic variants

# Preparing data for NLP analysis
df_nlp_pat = prep_nlp(final_data_pat, set_diff_)
df_nlp_pat["Mutation"]
corpus_pat = []

for i in df_nlp_pat["Mutation"]:
    corpus_pat.append(i)

# Saving variant files for samples separately 
saving_mut_files(df_nlp, "_pat_updated")

###############################################################################
```


```python
### Creating object for NLP for pathogenic variants 

# Counting the number of pathogenic variants that are identified in each sample.
df_nlp_pat["Mutation"].apply(lambda x: len(str(x).split(", "))).sum()

# Calculating Cosine similarities.
co_sim1_df_pat, co_sim1_doc = Cosine_similarity(corpus=corpus_pat)

# Getting the ID of the samples as index and column names. 
id_names = df_nlp_pat.Samplename
id_names = id_names.rename("id_names")
col_names = df_nlp_pat.Samplename
col_names = col_names.rename("col_names")

# Renameing the index and the columns.
co_sim1_df_pat.index = id_names
co_sim1_df_pat.columns = col_names

co_sim1_df_pat.columns = [col.replace("T", "TT") for col in co_sim1_df_pat.columns]
co_sim1_df_pat.index = [idx.replace("T", "TT") for idx in co_sim1_df_pat.index]

# Plotting the clustermap
plt.figure()
sns.clustermap(co_sim1_df_pat, cmap="Reds", yticklabels=True, xticklabels=True, annot=False, figsize=(12,12), dendrogram_ratio=0.15)

# Saving the plot
#save_fig(plt, "clustermap_pat.svg", False)

###############################################################################
```


```python
### Preparing for network analysis for pathogenic variants
  
# applying the product method
print(list(product(co_sim1_df_pat.index, co_sim1_df_pat.columns)))

# Creating series for input for the network analysis
sources_pat, targets_pat, weights_pat = source_targets_weights(co_sim1_df_pat)

###############################################################################
```


```python
### Creating and plotting the network with cosine similarity as weights for pathogenic variants. 

# Seed must be run in the same cell for reproducibility
seed = 124
random.seed(seed)
np.random.seed(seed)

# Plotting the network
plot_network(sources = sources_pat, targets = targets_pat, weights = weights_pat)

# Saving the plot
#save_fig(plt, "network_pat_updated.svg", False)

###############################################################################
```


```python
### Preparing data for co-mutational plotting for pathogenic variants

# Grouping data based on their sample ID. 
o_final_data = final_data_pat.copy()

add_final_data = pd.DataFrame(index=set_diff_, columns=o_final_data.columns)
add_final_data["samplename"] = list(set_diff_)
add_final_data["samplename_alias"] = ["Tissue"+ x[-3:] if "600006" in x else "Ascites"+ x[-3:] for x in add_final_data["samplename"]]
add_final_data["sample_ID"] = add_final_data["samplename"].apply(lambda x: x[-3:])
final_data_pat["sample_ID"] = final_data_pat["sample_ID"].apply(lambda x: x.replace("_", ""))
add_final_data["biopsy_type_"] = add_final_data["samplename_alias"].apply(lambda x: x[:-3])
add_final_data.reset_index(drop=True, inplace=True)
add_final_data.fillna("[]", inplace=True)
# add_final_data["sample_ID"] = [x.replace("_", "") for x in add_final_data.sample_ID]
# add_final_data["sample_ID_"] = add_final_data.biopsy_type_.apply(lambda x: x[:1])
# add_final_data["sample_ID"] = add_final_data["sample_ID_"] + add_final_data["sample_ID"]

# Concatenating data
o_final_data = pd.concat([o_final_data, add_final_data])

clinical_data_ = pd.read_excel(r"H:\PhD\Work_Packages\Work_package1_Ascites\Kliniske_data\DGCD_data_sampling.xlsx")
o_final_data["samplename"] = o_final_data.samplename.apply(lambda x: x.replace(".", "_"))
o_final_data["sample_ID"] = o_final_data.sample_ID.apply(lambda x: x.replace("_", "")).astype("int64")

o_final_data = pd.merge(o_final_data, clinical_data_, left_on="sample_ID", right_on="ID", how="inner")
o_final_data = o_final_data.replace("[]", np.nan)

###############################################################################
```


```python
### Comutational plotting

# Mutational data
sample_data = pd.DataFrame()
sample_data["sample"] = o_final_data["samplename_alias"]
sample_data["category"] = o_final_data["Genes1"]
sample_data["value"] = o_final_data["Variant Effect"]
sample_data["Histology"] = o_final_data["Histology_"]
sample_data["grade"] = o_final_data["GRAD"]
# sample_data["stage"] = o_final_data["Stage_OC_I_IV"]

# sorting samples 
cancer_order = ["HGSC", "LGSC", "Endometrioid", "Carcinosarcoma", "Mucinous","Neuroendocrine carcinoma", "Carcinoma", "Teratome", "Unknown"]

df_mapping = pd.DataFrame({
    'Histology': cancer_order,
})
sort_mapping = df_mapping.reset_index().set_index('Histology')

sample_data["num"] = sample_data["Histology"].map(sort_mapping["index"])
sample_data["ID"] = sample_data["sample"].apply(lambda x: x[-2:]).astype("int64")

sample_data = sample_data.sort_values(by="num")

# Saving data
# sample_data.to_excel(r"sample_data.xlsx")

sample_data = pd.read_excel(r"H:\PhD\Work_Packages\Work_package1_Ascites\Data\sample_data_.xlsx")
sample_data["sample"] = ["T"+ x[-2:] if x.startswith("T") else "A"+x[-2:] for x in sample_data["sample"]]

# Using groupby & sort_values to sort.
sample_data["ID"] = sample_data.ID.astype("str")

final_data["sample_ID"] = final_data.sample_ID.apply(lambda x: x.replace("_", "")).astype("str")
sample_data = sample_data.loc[sample_data["ID"].isin(final_data.sample_ID)]

sample_data = sample_data.reindex(index =  [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
                                          17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
                                          34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50,
                                          51, 52, 53, 70, 71, 72, 73, 74, 75, 78, 79, 80, 81, 82, 83, 90, 91, 92, 84, 85,
                                          86, 87])
# Defining category order of genes
category_order = ['CTNNB1','PTEN','FGFR2','MSH6','ATM','BRCA2','NF1','PIK3CA','KRAS','TP53']

# Creating dictionary with colors and params
mut_mapping = {'missense': '#00A893', 
               "nonsense": '#986B8C', 
               "frameshiftDeletion" : "#F6931E", 
               np.nan:"grey",
               'unknown' : '#0b819c',
               'Absent': {'facecolor': 'grey', 'alpha': 0.2}}

###############################################################################
```


```python
### Creating indicator element

indicator_df = pd.DataFrame()
indicator_df["sample"] = sample_data["sample"].unique()
indicator_df["group"] = indicator_df["sample"].apply(lambda x: x[-2:]).astype("int64")
indicator_df["category"] = "Same patient"

###############################################################################
```


```python
### Creating Cancer type element

cancer_type = sample_data[["sample", "Histology"]]
cancer_type = cancer_type.rename(columns = {"Histology":"value"})
cancer_type["category"] = "Histology"
cancer_type = cancer_type.drop_duplicates(subset=["sample"])

###############################################################################
```


```python
### Creating side-bar chart element
mut_freq = sample_data["category"].value_counts()

mut_freq = mut_freq.reset_index()
mut_freq = mut_freq.rename(columns={"index":"category",
                                    "category":"Mutated samples"})

side_mapping = {'Mutated samples': 'darkgrey'}

# calculate the percentage of samples with that gene mutated, rounding and adding a percent sign
percentages = (mut_freq['Mutated samples']/86*100).round(1).astype(str) + '%'


# decrease size of side bar
widths = [0.4, 4.5]

# move side bar plot slightly closer to comut
wspace = 0.10

###############################################################################
```


```python
### Creating grade element

grade = sample_data[["sample", "grade"]]
grade = grade.drop_duplicates(subset="sample")

grading = pd.get_dummies(grade.grade)
grading = grading.rename(columns={0:"Unknown",
                                  1:"Grade_1",
                                  2:"Grade_2",
                                  3:"Grade_3",
                                  4:"Grade_4"})

grading["sample"] = grade["sample"]
grading["Grade_4"] = [4 if x == 1 else x for x in grading["Grade_4"]]
grading["Grade_3"] = [3 if x == 1 else x for x in grading["Grade_3"]]
grading["Grade_2"] = [2 if x == 1 else x for x in grading["Grade_2"]]
grading["Grade_1"] = [1 if x == 1 else x for x in grading["Grade_1"]]
grading["Unknown"] = [0 if x == 1 else x for x in grading["Unknown"]]

grading = grading[["sample", "Unknown", "Grade_1", "Grade_2", "Grade_3", "Grade_4"]]

bar_mapping = {"Grade_4": '#be2b9e', "Grade_3": '#7a3cae', "Grade_2" : '#dbdad9', "Grade_1": '#97a191', "Unknown":'white'}
bar_kwargs = {'width': 0.8, 'edgecolor': 'black'}

###############################################################################
```


```python
### Creating Stage element

stage = pd.DataFrame()
stage["sample"] = o_final_data["samplename_alias"]
stage["sample"] = ["T"+ x[-2:] if x.startswith("T") else "A"+x[-2:] for x in stage["sample"]]


stage["stage"] = o_final_data["Stage_OC_I_IV"]
stage["Stage_4"] = [4 if x == "IV" else 0 for x in stage["stage"]]
stage["Stage_3"] = [3 if x == "III" else 0 for x in stage["stage"]]
stage["Stage_2"] = [2 if x == "II" else 0 for x in stage["stage"]]
stage["Stage_1"] = [1 if x == "I" else 0 for x in stage["stage"]]
stage["Unknown"] = [0 if x == np.nan else 0 for x in stage["stage"]]

stage = stage.drop_duplicates(subset=["sample"])
stage = stage.drop(columns=["stage"])

bar_mapping_stage = {"Stage_4": '#476930', "Stage_3": '#86B049', "Stage_2" : '#C8B88A', "Stage_1": '#F1DDDF', "Unknown":'white'}
bar_kwargs = {'width': 0.8, 'edgecolor': 'black'}

###############################################################################
```


```python
### Creating Overall survival element

overall_survival = pd.DataFrame()
overall_survival["sample"] = o_final_data["samplename_alias"]
overall_survival["sample"] = ["T"+ x[-2:] if x.startswith("T") else "A"+x[-2:] for x in overall_survival["sample"]]

overall_survival["survival"] = o_final_data["survival"]
overall_survival["survival"].fillna(0, inplace=True)

overall_survival = overall_survival.drop_duplicates(subset=["sample"])
bar_mapping_surv = {"survival": '#A62C2B'}
bar_kwargs = {'width': 0.8, 'edgecolor': 'black'}

###############################################################################
```


```python
### Adjusting space between elements

# decrease space between plots with space and wspace
hspace = 0.08
wspace = 0.1

# increase height of biopsy site from default 1 and decrease burden height from default 3
heights = {'Histology': 2, "Same patient" : 2}

structure = [['Mutation type'],['Same patient'], ["Histology"], ["Tumor Grade"]]

# decrease distance between purity and biopsy site
subplot_hspace = 0.02

###############################################################################
```


```python
### Plotting the co-mutational plot

sns.set_theme(style="white")

ova_comut = comut.CoMut()

# define order of comut BEFORE any data is added
# ova_comut.samples = sample_order

ova_comut.add_categorical_data(sample_data, name="Mutation type", 
                             mapping=mut_mapping, 
                             category_order=category_order,
                             tick_style="italic")
ova_comut.add_sample_indicators(indicator_df, name = "Same patient")
ova_comut.add_categorical_data(cancer_type, name="Histology", value_order=cancer_order)
ova_comut.add_side_bar_data(mut_freq, paired_name = 'Mutation type', name = 'Mutated samples', position = 'left', 
                            mapping = side_mapping, xlabel = 'Mutated samples', bar_kwargs = {'alpha': 0.5})
ova_comut.add_bar_data(grading, name = 'Tumor Grade', ylabel="Grade", mapping=bar_mapping, stacked=True, bar_kwargs=bar_kwargs)
ova_comut.add_bar_data(stage, name = 'Tumor Stage', ylabel="Stage", mapping=bar_mapping_stage, stacked=True, bar_kwargs=bar_kwargs)
ova_comut.add_bar_data(overall_survival, name = 'Survival', ylabel="Survival \n (mth)", mapping=bar_mapping_surv, stacked=True, bar_kwargs=bar_kwargs)


ova_comut.plot_comut(figsize = (20, 6), x_padding = 0.04, y_padding = 0.04, tri_padding = 0.03, widths = widths, wspace = wspace, hspace=hspace)#, structure=structure)
ova_comut.add_unified_legend(rename = {'Absent': 'Wild type'}, ncol=2)

# Saving the plot
#ova_comut.figure.savefig('ova_comut_updated.svg', bbox_inches = 'tight', dpi=600)

###############################################################################
```
