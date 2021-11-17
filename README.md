# EAT
Embedding-based annotation transfer (EAT) uses Euclidean distance between vector representations (embeddings) of proteins to transfer annotations from a set of labeled lookup protein embeddings to query protein embeddings.


# Abstract 
Here, we present a novel approach that expands the concept of homology-based inference (HBI) from a low-dimensional sequence-distance lookup to the level of a high-dimensional embedding-based annotation transfer (EAT). More specifically, we replace sequence similarity as means to transfer annotations from one set of proteins (lookup; usually labeled) to another set of proteins (queries; usually unlabeled) by Euclidean distance between single protein sequence representations (embeddings) from protein Language Models (pLMs). Secondly, we introduce a novel set of embeddings (dubbed ProtTucker) that were optimized towards constraints captured by hierarchical classifications of protein 3D structures (CATH). These new embeddings enabled the intrusion into the midnight zone of protein comparisons, i.e., the region in which the level of pairwise sequence similarity is akin of random relations and therefore is hard to navigate by HBI methods. Cautious benchmarking showed that ProtTucker reached further than advanced sequence comparisons without the need to compute alignments allowing it to be orders of magnitude faster.

# Getting started
Install bio_embeddings package as described here: https://github.com/sacdallago/bio_embeddings


Clone the EAT repository and get started as described in the Usage section below:
   ```sh
   git clone https://github.com/Rostlab/EAT.git
   ```

# Usage
- Quick start: General purpose (1 nearest neighbor/1-NN) without additional labels files:

For general annotation transfer/nearest-neighbor search in embedding space, the pLM ProtT5 is used. It was only optimized using raw protein sequences (self-supervised pre-training) and is therefor not biased towards a certain task. The following command will take two FASTA files holding protein sequences as input (lookup & queries) in order to transfer annotations (fasta headers) from lookup to queries:

```sh
python eat.py --lookup data/example_data_subcell/deeploc_lookup.fasta --queries data/example_data/la_query_setHARD.fasta --output eat_results/
```
- Extended: General purpose (3-NN) with additional labels:

If you want to provide your labels as separate file (labels are expected to have CSV format with 1st col being the fasta header and 2nd col being the label) and retrieve the first 3 nearest-neighbors (NN) instead of only the single NN:


```sh
python eat.py --lookup data/example_data_subcell/deeploc_lookup.fasta --queries data/example_data/la_query_setHARD.fasta --output eat_results/ --lookupLabels data/example_data_subcell/deeploc_lookup_labels.txt --queryLabels data/example_data_subcell/la_query_setHARD_labels.txt
```
Example output is given here: [Example output](https://github.com/Rostlab/EAT/blob/main/data/example_data_subcell/example_output_protT5_NN3.txt)

- Expert solution tailored for remote homology detection:

For remote homology detection, we recommend to use ProtTucker(ProtT5) embeddings that were specialized on capturing the CATH hierarchy:

```sh
python eat.py --lookup data/example_data_subcell/deeploc_lookup.fasta --queries data/example_data/la_query_setHARD.fasta --output eat_results/ --use_tucker 1
```

# Pre-computed lookup embeddings
We have pre-computed embeddings for major databases to simplify your EAT search (release date of DBs: 16.11.2021):

- SwissProt (565k proteins, 1.3GB): [Download SwissProt embeddings](https://rostlab.org/~deepppi/eat_dbs/sprot_161121.h5) 
- PDB (668k chains, 1.6GB): [Download PDB embeddings](https://rostlab.org/~deepppi/eat_dbs/pdb_seqres_161121.h5)
- CATH-S100 v4.3 (122k chains, 286MB): [Download CATH embeddings](https://rostlab.org/~deepppi/eat_dbs/cath_v430_dom_seqs_S100_161121.h5)
- SCOPe v.2.08 (93k chains, 216MB): [Download SCOPe embeddings](https://rostlab.org/~deepppi/eat_dbs/scope_2.08_S100.h5)

All embeddings listed above were generated using ProtT5-XL-U50 (or in short ProtT5).

In a first step, per-residue embeddings (Lx1024 for ProtT5) were computed. 
Per-protein embeddings were derived by averaging over the per-residue embeddings, resulting in a single 1024-d vector for each protein, irrespective of its length.
Embeddings are stored as H5 files with protein/chain identifiers (either SwissProt-, PDB-, CATH-, or SCOPe-IDs) as keys and 1024-d embeddings as values.

The model was run in half-precision mode on a Quadro RTX 8000 with 48GB vRAM. 
Proteins longer than 9.1k residues had to be excluded due to OOM-errors (only a handful proteins were affected by this).

The embeddings can readily be used as input to the lookup file parameter of EAT.
IF you want to detect structural homologs, we recommend running the provided lookup embeddings through ProtTucker(ProtT5) first. This can be done by adding the flag `--use_tucker 1`.

# Reference (Pre-print)
```
@article {Heinzinger2021.11.14.468528,
	author = {Heinzinger, Michael and Littmann, Maria and Sillitoe, Ian and Bordin, Nicola and Orengo, Christine and Rost, Burkhard},
	title = {Contrastive learning on protein embeddings enlightens midnight zone at lightning speed},
	year = {2021},
	doi = {10.1101/2021.11.14.468528},
	URL = {https://www.biorxiv.org/content/early/2021/11/15/2021.11.14.468528},
	journal = {bioRxiv}
}
```
