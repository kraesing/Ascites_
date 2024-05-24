#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:39:19 2023

@author: LVES0067
"""
# =============================================================================
#         			Functions
# =============================================================================

###############################################################################
#%% IMPORTING LIBRARIES

# Libraries for file management
import os
import glob
import fnmatch
import shutil #https://docs.python.org/3/library/shutil.html

# Libraries for data management
import pandas as pd 
import numpy as np

# Libraries for statistical analysis
import scipy.stats as stats
from scipy.spatial.distance import squareform, pdist

# Libraries for plotting.
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib_venn import venn2, venn2_circles, venn2_unweighted, venn3, venn3_circles

# Setting params for plotting
sns.set_theme(style="whitegrid")
sns.set_context("notebook")
plt.rcParams["font.family"] = "times new roman"
sns.set_palette("colorblind")
sns.set_color_codes(palette="muted")
import matplotlib.lines as mlines

# For machine learning
import tensorflow as tf

# Dimensional reduction
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore")

# Natural Language Processing
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# FOR NETWORKS
import networkx as nx
from pyvis.network import Network


###############################################################################
#%% CODE AWAY
print("Libraries imported - Code away!")

###############################################################################
#%% CHANGE DIRECTORY
path = "H:/PhD/Work_Packages/Work_package1_Ascites/Data/"

os.chdir(path + "IonTorrent5.18")

# Setting up environment
if os.path.exists("figures"):
    print("Directory for figures already exist!")
else:
    os.mkdir("figures")
    print("Directory for figures has been created!")

###############################################################################
#%% FUNCTION FOR SAVING FIGURES 


def save_fig(plot_name ,filename, transparent_background):
    os.chdir(path + "figures")
    plot_name.savefig(filename, format="svg", dpi=600, bbox_inches='tight', 
                      transparent=transparent_background)
    print(f"Your figure: {filename}, has been saved!")
    os.chdir(path)
    
###############################################################################
#%% FUNCTION FOR COUNTING VARIANTS

def counting_variants(input_list1, input_list2):
    variant_counts1 = []
    for i in input_list1:
        variant_counts1.append(pd.read_csv(i, delimiter="\t").assign(biopsy_type="ascites"))
    variant_counts2 = []
    for i in input_list2:
        variant_counts2.append(pd.read_csv(i, delimiter="\t").assign(biopsy_type="tissue"))
    
    df_combined = pd.concat(variant_counts1 + variant_counts2)
    print("total number of variants:", len(df_combined))
    print(df_combined.biopsy_type.value_counts())
    print(df_combined.Genes.nunique())

###############################################################################
#%% FUNCTION FOR INITIAL CLEANING 

df_list = []
list_nocall = []
df_filtered = []

def Initial_cleaning(df):
    List_of_types = ["SNV", "INDEL", "MNV"]
    list_nocall.append(df.loc[df["Filter"] == "NOCALL"])
    df = df.loc[df["Type"].isin(List_of_types)]
    df.dropna(how="all", axis=1, inplace=True)
    list_to_remove = []
    for col in df.columns:
        if "ExAC" in col:
            list_to_remove.append(col)
    df = df.drop(columns=list_to_remove)
    #df = df.drop(columns=(["SIFT", "Grantham", "PolyPhen", "PhyloP", "DrugBank"]))
    df["Chrom"] = df["Locus"].str.split(":").str[0]
    df["Start"] = pd.to_numeric(df["Locus"].str.split(":").str[1])
    df["End"] = df["Start"] + df["Length"] - 1
    df["Base_call_accuracy"] = 1-10**-(df["Phred QUAL Score"]/10)
    list_of_allele_ratios = list(df["Allele Ratio"].str.findall("\d+\.?\d+"))
    df[["Allele_Ratio1", "Allele_Ratio2"]] = list_of_allele_ratios
    df[["Allele_Ratio1", "Allele_Ratio2"]] = df[["Allele_Ratio1", "Allele_Ratio2"]].apply(pd.to_numeric, axis=1)
    df["Potential_germline"] = df["Allele_Ratio2"].apply(lambda x: "Potential germline variant" if x >= 0.98 else "NaN")
    df = df.loc[df["Potential_germline"] != "Potential germline variant"]
    df = df.loc[df["UCSC Common SNPs"] != "YES"]
    df = df.loc[df["Variant Effect"] != "synonymous"]
    df = df.loc[(df["Location"].str.contains("exonic", na=False)) | (df["Location"].str.contains("splicesite", na=False))]
    
    return df

###############################################################################
#%% FUNCTIONS FOR GENE CLEANING

def gene_cleaning():
    df_OCA_bed = pd.read_csv(r"OCAv3.20180509.Designed.bed", sep="\t", skiprows=(1), header=None)
    df_OCA_bed = df_OCA_bed.set_axis(["Chr", "Start", "End", "primer_ID", "seperator", "Gene_information"], axis=1)

    df_OCAPlus_bed = pd.read_csv(r"OCAPlus.20191203.designed.bed", sep="\t", skiprows=(1), header=None)
    df_OCAPlus_bed = df_OCAPlus_bed.set_axis(["Chr", "Start", "End", "primer_ID", "seperator", "Gene_information"], axis=1)

    df_OCA_bed["Genes"] = df_OCA_bed["Gene_information"].str.split(";").str[0].str.split("=").str[1]
    OCA_genes = set(df_OCA_bed["Genes"].unique().tolist())

    df_OCAPlus_bed["Genes"] = df_OCAPlus_bed["Gene_information"].str.split(";").str[0].str.split("=").str[1]
    df_OCAPlus_bed = df_OCAPlus_bed[~df_OCAPlus_bed.Genes.str.contains("rs")]
    OCAPlus_genes = set(df_OCAPlus_bed["Genes"].unique().tolist())

    Genes_int = set(OCA_genes.intersection(OCAPlus_genes))
    Unique_genes_OCA = Genes_int.symmetric_difference(OCA_genes)
    Unique_genes_OCAPlus = OCAPlus_genes.symmetric_difference(Genes_int)

    OCP_Start_end = df_OCA_bed[["Start", "End", "Genes"]]
    OCPPlus_Start_end = df_OCAPlus_bed[["Start", "End", "Genes"]]

    OCP_Start_end["Covered_positions"] = OCP_Start_end.apply(lambda x : list(range(x["Start"], x["End"]+1)),1)
    OCPPlus_Start_end["Covered_positions"] = OCPPlus_Start_end.apply(lambda x : list(range(x["Start"], x["End"]+1)),1)

    OCP_Start_end1 = OCP_Start_end.explode("Covered_positions")
    OCPPlus_Start_end1 = OCPPlus_Start_end.explode("Covered_positions")

    int_pos = set(set(OCP_Start_end1["Covered_positions"]).intersection(set(OCPPlus_Start_end1["Covered_positions"])))
    
    return Genes_int, int_pos

genes_int, int_pos = gene_cleaning()

def Gene_cleaning(df):
    df = df.assign(Genes1=df["Genes"].str.split(",")).explode("Genes1")
    df = df.loc[(df["Genes1"].isin(genes_int)) & (df["Start"].isin(int_pos))]
    
    return df

###############################################################################
#%% FUNCTION FOR ORIGINAL FILTERING OF VARIANTS

def Original_filtering(df, allele_freq, coverage):                                          
    
    df.loc[df["Allele Frequency %"] < allele_freq, "true_variant"] = "Below threshold for allele frequency"
    
    # Assigning variants with "low overall coverage" - introducing penalties for higher mapd. 
    
    df.loc[df["Coverage"] < (coverage), "true_variant"] = "Low overall coverage (caution warrant)"                                                                  

    # Assigning variants with "high polymer content"
    df.loc[(df["Homopolymer Length"] >= 5), "true_variant"] = "High homopolymer content"                                          

    # Transforming the mean Coverage pr group 
    #df["Mean_Coverage"] = df.groupby(["samplename"]).Coverage.transform("mean")                                                   

    # Transforming the mean of the allele_ratio_sum pr group
    #df["Mean_Allele_ratio_sum"] = df.groupby(["SampleName"]).Allele_Ratio2.transform("mean")                                   
    
    # Assignign variants with "Allele Ratio below Q1"
    #df.loc[df["Allele_Ratio2"] < 0.25*(df["Mean_Allele_ratio_sum"]), "True_Variant"] = "Allele Ratio below Q1"                  

    # df.loc[(df["Allele_Ratio2"] < 0.25*(df["Mean_Allele_ratio_sum"])) 
    #      & (df["Coverage"] > 0.5*(df["Mean_Coverage"])) 
    #      & (df["Base_call_accuracy"] > 0.99) 
    #      & (df["Homopolymer Length"] <= 4), "True_Variant"] = "Allele ratio below Q1, but high quality" 
      
    m = df.loc[lambda x: x["Coverage"] >= 100]                                                                                           
    m["Coverage >= 100"] = m.groupby(["samplename"]).Coverage.transform("mean")                                                 
    m1 = m[["samplename", "Coverage >= 100"]].groupby("samplename").first().reset_index()                                       
    df = df.merge(m1, on="samplename", how="left")                                                                                
    df.loc[df["Coverage"] < 0.10*(df["Coverage >= 100"]), "true_variant"] = "Low base coverage"                                    
    df.loc[df["Phred QUAL Score"] < 200, "true_variant"] = "Low Phred Score"                                                      
    df.loc[df["P-Value"] > 0.01, "true_variant"] = "Above p-value"                                                                
    df["true_variant"] = df["true_variant"].fillna("PASS")

    return df

###############################################################################
#%% FUCNTION FOR RESCUE FILTERING, NO CALL VARIANTS. 
#   This filtering is based on observation from Vestergaard et al. 2021 OCAv3 vs OCAP paper.

def NOCALL_recue(df, Allele_ratio, P_value, Phredscore):
    NOCALL = df[df["Type"] == "NOCALL"]
    NOCALL["Allele_Ratio1"] = NOCALL["Allele Ratio"].str.split(",").str[0].str.split("=").str[1].astype("float")
    NOCALL["Allele_Ratio2"] = NOCALL["Allele Ratio"].str.split(",").str[1].str.split("=").str[1].astype("float")
    NOCALL.loc[NOCALL["Allele_Ratio2"] == 1, "Rescue_filtering"] = "Germline/NoVariant"
    NOCALL.loc[NOCALL["Allele_Ratio2"] < Allele_ratio, "Rescue_filtering"] = "Allele ratio below %f" % (Allele_ratio)
    NOCALL.loc[NOCALL["P-Value"] > P_value, "Rescue_filtering"] = "P-value above %f" % (P_value)
    NOCALL.loc[NOCALL["Phred QUAL Score"] < Phredscore, "Rescue_filtering"] = "Phred score under criteria"
    NOCALL.loc[NOCALL["UCSC Common SNPs"] == "YES", "Rescue_filtering"] = "Common SNP"
    NOCALL.loc[NOCALL["Homopolymer Length"] >= 5, "Rescue_filtering"] = "High Homopolymer content"
    NOCALL.loc[NOCALL["Rescue_filtering"].isnull(), "Rescue_filtering"] = "PASS NOCALL"
    NOCALL = NOCALL.loc[NOCALL["Rescue_filtering"] == "PASS NOCALL"]
    print(NOCALL["Rescue_filtering"].value_counts())
    
    return NOCALL

###############################################################################
#%% FUNCTION FOR RESCUE FILTERING, RESCUE CLEANING.

def Variant_rescue_cleaning(df):
    df.loc[df["UCSC Common SNPs"] == "YES", "Rescue_cleaning"] = "Common SNP"
    df.loc[df["Variant Effect"] == "synonymous", "Rescue_cleaning"] = "synonymous mutation" 
    df.loc[df["P-Value"].isnull(), "Rescue_cleaning"] = "missing p_value"
    # df.loc[df["ClinVar"] == "Benign", "Rescue_cleaning"] = "Benign variant"
    df.loc[df["Homopolymer Length"] >= 5, "Rescue_cleaning"] = "High Homopolymer content"
    df.loc[df["Rescue_cleaning"].isnull(), "Rescue_cleaning"] = "PASS Rescue cleaning"
    # print(df["Rescue_cleaning"].value_counts())
    
    df = df.loc[df["Rescue_cleaning"] == "PASS Rescue cleaning"]

    return df

###############################################################################
#%% FUNCTION FOR RESCUE FILTERING.

def Variant_rescue_filtering(df, Frequency, Pvalue, PhredScore, Coverage, suffix):
    df.loc[df["Allele Frequency %"] < Frequency, "Rescue_filtering"] = "Allele Ratio below 2.5%"
    df.loc[df["P-Value"] > Pvalue, "Rescue_filtering"] = "P-value above criteria"
    df.loc[df["Phred QUAL Score"] < PhredScore, "Rescue_filtering"] = "Phred Score below criteria"
    df.loc[df["Raw Coverage"] < Coverage, "Rescue_filtering"] = "Coverage below criteria"
    df.loc[df["StrandBias"] == "Caution to strandbias", "Rescue_filtering"] = "Strandbias"
    df.loc[df["Rescue_filtering"].isnull(), "Rescue_filtering"] = "PASS Rescue filtering %s" % (suffix)
    
###############################################################################
#%% FUNDTION FOR CALCULATING STRANDBIAS ON IDENTIFIED VARIANTS.

def apply_fishers_test_(df, strandbias_threshold):
    lst_ref_var = list(df["Ref+/Ref-/Var+/Var-"].str.findall(r'\d+'))
    df[["Ref+", "Ref-", "Var+", "Var-"]] = lst_ref_var
    
    df["Oddsratio"], df["FisherE_Pvalue"] = zip(*df.apply(lambda r: stats.fisher_exact([[r["Ref+"], 
                                                                                         r["Ref-"]],
                                                                                        [r["Var+"], 
                                                                                         r["Var-"]]]), axis=1))
    
    df["StrandBias"] = df["FisherE_Pvalue"].apply(lambda x: "Caution to strandbias" if x <= strandbias_threshold else "No strandbias") # https://datatofish.com/if-condition-in-pandas-dataframe/    

    return lst_ref_var

###############################################################################
#%% DEFINING PREVIOUS ARTIFICIAL VARIANTS

artifactual_var = [["chr3:189456566", "TP63", "p.?", "Variant called on the very end of amplicon"],
                   ["chr12:133220098", "POLE", "p.Val1446GlyfsTer3", ""],
                   ["chr1:27100181", "ARID1A", "p.Gln1334del", "Variant called within highly repeated area of GCAGCAGCA"],
                   ["chr7:87258149", "ABCB1, RUNDC3B", "p.?, p.Arg4Trp", "Variant called within the first three nucleotides of the amplicon"],
                   ["chrX:44928852", "KDM6A", "p.Ser651Leu", "Variant identified from not full lenght amplicon"],
                   ["chrX:44928914", "KDM6A", "p.Ala672Ser", "Variant identified from not full lenght amplicon"],
                   ["chrX:44928920", "KDM6A", "p.Ser674Thr", "Variant identified from not full lenght amplicon"],
                   ["chr3:10183605", "VHL", "p.Pro25Leu", "Variant identified from not full lenght amplicon"],
                   ["chr7:151859709", "KMT2C", "p.Glu3651Asp", "Variant identified from not full lenght amplicon"],
                   ["chr8:128750685", "MYC", "p.[Pro74=;Pro75Ser]", "Variant identified from not full lenght amplicon"],
                   ["chr12:133212583", "POLE", "p.Ser1902PhefsTer3", "Variant only annotated in one direction"],
                   ["chr5:79950735", "DHFR,MSH3", "p.?, p.Ala61_Pro63dup", "Variant identified from not full lenght amplicon and only in one direction"], 
                   ["chr2:48026247", "MSH6", "p.Arg379Ter", "Variant called within repeated area"], 
                   ["chr1:120612040", "NOTCH2", "p.?, p.?", "Variant only in one direction"], 
                   ["chr1:120612037", "NOTCH2", "p.?, p.?", "Variant only in one direction"],
                   ["chr1:120612031", "NOTCH2", "p.?, p.?", "Variant only in one direction"],
                   ["chr1:120612039", "NOTCH2", "p.?, p.?", "Variant only in one direction"]]

df_artifactual_var = pd.DataFrame(artifactual_var, columns =["Locus", "Gene", "Amino_Acid_Change", "Verdict"])

############################################################################### 
# #%% Filter out artifacts
def remove_artifactual_variants(df):
    global df_artifactual_var
    
    df.loc[(df["Locus"].isin(df_artifactual_var["Locus"])) & 
           (df["Amino Acid Change"].isin(df_artifactual_var["Amino_Acid_Change"])),
           "Common SNP"] = "Not common SNP, artifactual variant"
    
    df.loc[df["Common SNP"].isnull(), "Common SNP"] = "Variant still of importance"
    
    df = df.loc[df["Common SNP"] == "Variant still of importance"]
    
    return df

list_benigns = ["Benign", "Benign/Likely benign", "Likely benign"]

############################################################################### 
#%% PEARSON CORRELATION 

def pearson_coeff(df, input1, input2):
    pearson_1 = set(df.loc[df["biopsy_type_"] == input1]["for_venn"])
    pearson_2 = set(df.loc[df["biopsy_type_"] == input2]["for_venn"])
    
    pearson_int = pearson_1.intersection(pearson_2)
    
    pearson_1 = df.loc[(df["biopsy_type_"] == input1) & (df["for_venn"].isin(pearson_int))][["for_venn", "Allele Frequency %"]]
    pearson_1.rename(columns={"Allele Frequency %" : "AF_{}".format(input1)}, inplace=True)
    
    pearson_2 = df.loc[(df["biopsy_type_"] == input2) & (df["for_venn"].isin(pearson_int))][["for_venn", "Allele Frequency %"]]
    pearson_2.rename(columns={"Allele Frequency %" : "AF_{}".format(input2)}, inplace=True)
    
    pearson_merged = pd.merge(pearson_2, pearson_1, on="for_venn")
    
    r, p = stats.pearsonr(pearson_merged["AF_{}".format(input2)], pearson_merged["AF_{}".format(input1)])
    r = round(r, 4)
    p = round(p, 4)
    
    return r, p, pearson_merged

###############################################################################
#%% PREPARING DATA FOR NLP

def prep_nlp(df, diff_samples):
    
    dict_mutation = {"Samplename" : [],
                     "Mutation" : []}
    
    for i in df["samplename"].unique():
        df_mutation = df.loc[df["samplename"] == i]
        
        l = [', '.join(df_mutation["mutation"])]
        
        dict_mutation["Samplename"].append(i)
        dict_mutation["Mutation"].append(l)
        
    # Create Dataframe from dict
    df_nlp = pd.DataFrame.from_dict(dict_mutation)
    
    for i in diff_samples:
        df_nlp = df_nlp.append({"Samplename" : i, "Mutation": []}, ignore_index=True)
        
    df_nlp["Mutation"] = df_nlp["Mutation"].apply(lambda x: str(x).replace("[", ""))
    df_nlp["Mutation"] = df_nlp["Mutation"].apply(lambda x: str(x).replace("]", ""))
    df_nlp["Mutation"] = df_nlp["Mutation"].apply(lambda x: str(x).replace(".", "_"))
    df_nlp["Mutation"] = df_nlp["Mutation"].apply(lambda x: str(x).replace(":", "_"))
    df_nlp["Mutation"] = df_nlp.Mutation.astype("str")
    df_nlp["Mutation"] = df_nlp["Mutation"].apply(lambda x: str(x).replace("'", ""))
    df_nlp["Samplename"] = df_nlp.Samplename.apply(lambda x: x.replace(".", "_"))
    df_nlp["Samplename"] = df_nlp.Samplename.apply(lambda x: x.replace("_", ""))
    df_nlp["Samplename"] = df_nlp.Samplename.apply(lambda x: x[-5:])
    df_nlp["Samplename"] = [x.replace("006", "T") if x.startswith("006") else x.replace("005", "A") for x in df_nlp.Samplename]

    return df_nlp

############################################################################### 
#%% SAVING MUTATION FILES SEPERATELY
def saving_mut_files(df, suffix):

    for i,j in zip(df.Samplename, df.Mutation):

        file_name = i+suffix+".txt"
    
        text_file = open(file_name, "w")
        text_file.write(j)
        text_file.close()

############################################################################### 
# %% COSINE SIMILARITY

def Cosine_similarity(corpus):
    
    #Instantiate a count vector
    count_vect = CountVectorizer()
  
    # Train the Count Vectorizer.
    X_train_counts = count_vect.fit_transform(corpus)
    
    # Converting the X_train_counts to a dataframe. 
    df_doc = pd.DataFrame(X_train_counts.toarray(), columns=count_vect.get_feature_names())
    
    # Calculate cosine similarity
    co_sim = cosine_similarity(X_train_counts, X_train_counts)
    
    # Instantiate a TfidfVectorizer
    vectorizer_tfidf = TfidfVectorizer()

    # Fit and transform the data    
    trsfm = trsfm = vectorizer_tfidf.fit_transform(corpus)
    
    # Converting trsfm to a dataframe
    df_doc1 = pd.DataFrame(trsfm.toarray(), columns=count_vect.get_feature_names()) # Make principle component analysis on this. 

    # Calculating cosine similarity
    co_sim1 = cosine_similarity(trsfm, trsfm)

    # Creating a dataframe over the cosine similarities
    co_sim1_df = pd.DataFrame(co_sim1)
    
    return co_sim1_df, df_doc1

############################################################################### 
#%% PLOTTING NETWORKS

def plot_network(sources, targets, weights):

    plt.figure(figsize=(20, 20))

    # Create directed graph object
    Graph = nx.Graph()

    # https://networkx.org/documentation/stable/auto_examples/drawing/plot_weighted_graph.html

    for s,t,w in zip(sources, targets, weights.round(2)):
        Graph.add_edge(s, t, weight=w, value=w)

    exlarge = [(u, v) for (u, v, d) in Graph.edges(data=True) if d["weight"] > 0.90]
    elarge = [(u, v) for (u, v, d) in Graph.edges(data=True) if d["weight"] > 0.75 and d["weight"] < 0.90]
    emedium = [(u, v) for (u, v, d) in Graph.edges(data=True) if d["weight"] > 0.50 and d["weight"] < 0.75]
    esmall = [(u, v) for (u, v, d) in Graph.edges(data=True) if d["weight"] < 0.50]

    pos = nx.spring_layout(Graph, seed=17, dim=2, iterations=40)
    nx.draw_networkx_edges(Graph, pos, edgelist=esmall, width=2, alpha=0.05, edge_color="gray", style="dashed")
    nx.draw_networkx_edges(Graph, pos, edgelist=exlarge, width=10, alpha=0.7, edge_color="#9F0E14", style="solid")
    nx.draw_networkx_edges(Graph, pos, edgelist=elarge, width=8, alpha=0.7, edge_color="#E43027", style="solid")
    nx.draw_networkx_edges(Graph, pos, edgelist=emedium, width=4, alpha=0.7, edge_color="#FB7050", style="solid")

    # node labels
    nx.draw_networkx_labels(Graph, pos, font_size=20, font_family="sans-serif")

    
    ax = plt.gca()
    ax.margins(0.08)
    plt.axis("off")
    plt.tight_layout()
    
    return plt

############################################################################### 
# %% PLOTTING CLUSTERMAP

def plotting_clustermap(df):

    plt.figure()
    sns.set_theme(style="white")
    g = sns.clustermap(df, yticklabels=True, xticklabels=True,
                       figsize=(35, 35), cmap="vlag", annot=False)

    mask = np.tril(np.ones_like(df))
    mask_inverted = np.logical_not(mask).astype(int)

    values = g.ax_heatmap.collections[0].get_array().reshape(df.shape)
    new_values = np.ma.array(values, mask=mask)
    new_values_inverted = np.ma.array(values, mask=mask)

    g.ax_heatmap.collections[0].set_array(new_values)
    g.ax_row_dendrogram.set_visible(False)
    
    plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=25)
    plt.setp(g.ax_heatmap.get_xticklabels(), fontsize=25)
    
    # plt.setp(g.ax_cbar.tick_params(labelsize=30))
    
###############################################################################     
# %% SOURCE TARGETS WEIGHTS FOR NETWORK

def source_targets_weights(df):
    
    # Getting all combinations of cosine similarities matrix in two dimensions
    t = df.stack()
    t = t.reset_index()
    t = t.rename(columns={"level_0": "node", "level_1": "connection", 0 : "cosine_weight"})
    
    # Remove samples that have the same node and connection, as they are the sample itself.
    t = t[t["node"] != t["connection"]]
    # t = t.drop(t[t["cosine_weight"] == 0].index)
    
    # Saving object into sources, targets and weights. 
    sources = t["node"]
    targets = t["connection"]
    weights = t["cosine_weight"]

    return sources, targets, weights

############################################################################### 
#%% PLOTTING SCATTER HEATMAP
def scatter_heatplot(data, x, y, hue, title):
    g = sns.relplot(
        data=data,
        x=x, y=y, hue=hue, size="size",
        palette="vlag", edgecolor="0.7",
        height=12, sizes=(50, 200), size_norm=(-0.01, 1.2), kind="scatter", 
        marker="o", hue_norm=(-1.3, 1.3), legend=True)
    
    # Tweak the figure to finalize
    g.set(xlabel="", ylabel="", aspect="equal")
    g.despine(left=False, bottom=False, top=False, right=False)
    g.ax.margins(.015)
    g.set_xlabels("Samples", size=20)
    g.set_ylabels("Samples", size=20)

    for label in g.ax.get_xticklabels():
        label.set_rotation(90)
        
    save_fig(g, "{}".format(title), transparent_background=True)








