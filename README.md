# EAT
Embedding-based annotation transfer (EAT) uses Euclidean distance between vector representations (embeddings) of proteins to transfer annotations from a set of labeled lookup protein embeddings to query protein embeddings.


# Abstract 
Here, we present a novel approach that expands the concept of homology-based inference (HBI) from a low-dimensional sequence-distance lookup to the level of a high-dimensional embedding-based annotation transfer (EAT). More specifically, we replace sequence similarity as means to transfer annotations from one set of proteins (lookup; usually labeled) to another set of proteins (queries; usually unlabeled) by Euclidean distance between single protein sequence representations (embeddings) from protein Language Models (pLMs). Secondly, we introduce a novel set of embeddings (dubbed ProtTucker) that were optimized towards constraints captured by hierarchical classifications of protein 3D structures (CATH). These new embeddings enabled the intrusion into the midnight zone of protein comparisons, i.e., the region in which the level of pairwise sequence similarity is akin of random relations and therefore is hard to navigate by HBI methods. Cautious benchmarking showed that ProtTucker reached further than advanced sequence comparisons without the need to compute alignments allowing it to be orders of magnitude faster.


# Usage
For general annotation transfer/nearest-neighbor search in embedding space, the pLM ProtT5 is used. It was only optimized using raw protein sequences (self-supervised pre-training) and is therefor not biased towards a certain task.


If one is interested in structural similarity/fold recognition, we recommend to use our proposed ProtTucker embeddings.



# Figures
<img src="https://github.com/Rostlab/EAT/blob/main/ProtTucker_tSNE.png?raw=true" width="60%" height="60%">
Contrastive learning improved CATH class-level clustering. Using t-SNE, we projected the high-dimensional embedding space onto 2D before (left) and after (right) contrastive learning. The colors mark the major class level of CATH (C) distinguishing proteins according to their major distinction in secondary structure content.

<br/><br/>


<img src="https://github.com/Rostlab/EAT/blob/main/ProtTucker_reliability.png?raw=true" width="50%" height="50%">
Similar to varying E-value cut-offs for homology-based inference (HBI), we examined whether the fraction of correct predictions (accuracy; left axis) depended on embedding distance (x-axis) for EAT. Toward this end, we transferred annotations for all four levels of CATH (Class: blue; Architecture: orange; Topology: green; Homologous superfamily: red) from proteins in our lookup set to the queries in our test set using the hit with smallest Euclidean distance. The fraction of test proteins having a hit below a certain distance threshold (coverage, right axis, dashed lines; Eqn. 3) was evaluated separately for each CATH level. For example, at a Euclidean distance of 1.1 (marked by black vertical dots), 78% of the test proteins found a hit at the H-level (Cov(H)=78%) and of 89% were correctly predicted (Acc(H)=89%). Similar to decreasing E-values for HBI, decreasing embedding distance correlated with EAT performance. This correlation importantly enables users to select only the, e.g., 10% top hits, or all hits with an accuracy above a certain threshold.


# Reference
tbd
