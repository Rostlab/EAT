#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:38:40 2021

@author: mheinzinger
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import h5py

import random
from sklearn.metrics import accuracy_score, balanced_accuracy_score,  f1_score


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Tucker(nn.Module):
    def __init__(self):
        super(Tucker, self).__init__()

        self.tucker = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
        )

    def single_pass(self, x):
        return self.tucker(x)

    def forward(self, X): # only needed during training
        ancor = self.single_pass(X[:, 0, :])
        pos   = self.single_pass(X[:, 1, :])
        neg   = self.single_pass(X[:, 2, :])
        return (ancor, pos, neg)


class Evaluator():
    def __init__(self, predictions):
        self.Ys, self.Yhats, self.reliabilities = zip(*[(query_label, lookup_label, eat_dist)
                                                        for _, query_label, _, lookup_label, eat_dist, nn_iter in predictions
                                                        if nn_iter == 0
                                                        ]
                                                      )

    def compute_performance(self):
        error_estimates = self.compute_err()
        for metric, (performance, bootstrap_err) in error_estimates.items():
            print("{}={:.3f} +/-{:.3f}".format(
                metric,
                performance,
                1.96*np.std(np.array(bootstrap_err), ddof=1)
            )
            )

        return None

    def compute_err(self, n_bootstrap=1000):

        n_total = len(self.Ys)  # total number of predictions
        idx_list = range(n_total)

        Ys, Yhats = np.array(self.Ys), np.array(self.Yhats)

        acc = accuracy_score(Ys, Yhats)
        f1 = f1_score(Ys, Yhats, average="weighted")
        bAcc = balanced_accuracy_score(Ys, Yhats)

        accs_btrap, f1s_btrap, bAccs_btrap = list(), list(), list()
        n_skipped = 0
        for _ in range(n_bootstrap):
            rnd_subset = random.choices(idx_list, k=n_total)
            # skip bootstrap iterations where predictions might hold labels not part of groundtruth
            if not set(Yhats[rnd_subset]).issubset(Ys[rnd_subset]):
                n_skipped += 1
                continue
            accs_btrap.append(accuracy_score(
                Ys[rnd_subset], Yhats[rnd_subset])
                )
            f1s_btrap.append(
                f1_score(Ys[rnd_subset], Yhats[rnd_subset], average="weighted")
                )
            bAccs_btrap.append(balanced_accuracy_score(
                Ys[rnd_subset], Yhats[rnd_subset])
                )

        print("Skipped {}/{} bootstrap iterations due to mismatch of Yhat and Y.".format(
            n_skipped, n_bootstrap))
        return {"ACCs": (acc, accs_btrap), "bACCs": (bAcc, bAccs_btrap), "F1": (f1, f1s_btrap)}


class Embedder():
    def __init__(self):
        from bio_embeddings.embed import ProtTransT5XLU50Embedder
        self.embedder = ProtTransT5XLU50Embedder(half_model=True)

    def write_embeddings(self, emb_p, embds):
        with h5py.File(str(emb_p), "w") as hf:
            for sequence_id, embedding in embds.items():
                # noinspection PyUnboundLocalVariable
                hf.create_dataset(sequence_id, data=embedding)
        return None

    def get_embeddings(self, id2seq):
        fasta_ids, seqs = zip(*[(fasta_id, seq)
                              for fasta_id, seq in id2seq.items()])
        print("Start generating embeddings. This process might take a few minutes.")
        start = time.time()
        per_residue_embeddings = list(self.embedder.embed_many(list(seqs)))
        id2embd = { fasta_id: per_residue_embeddings[idx].mean(axis=0)
                       for idx, fasta_id in enumerate(list(fasta_ids))
                   }
        print("Creating embeddings took: {:.4f}[s]".format(time.time()-start))
        return id2embd


# EAT: Embedding-based Annotation Transfer
class EAT():
    def __init__(self, lookup_p, query_p, output_d, use_tucker, num_NN,
                 lookupLabels, queryLabels):

        self.output_d = output_d
        Path.mkdir(output_d, exist_ok=True)
        
        self.num_NN = num_NN
        self.Embedder = None 
        
        self.lookup_ids, self.lookup_embs = self.read_inputs(lookup_p)
        self.query_ids, self.query_embs = self.read_inputs(query_p)

        if use_tucker:  # create ProtTucker(ProtT5) embeddings
            self.lookup_embs = self.tucker_embeddings(self.lookup_embs)
            self.query_embs = self.tucker_embeddings(self.query_embs)

        self.lookupLabels = self.read_label_mapping(
            self.lookup_ids, lookupLabels)
        self.queryLabels = self.read_label_mapping(self.query_ids, queryLabels)

    def tucker_embeddings(self, dataset):
        weights_p = self.output_d / "tucker_weights.pt"

        # if no pre-trained model is available, yet --> download it
        if not weights_p.exists():
            import urllib.request
            print(
                "No existing model found. Start downloading pre-trained ProtTucker(ProtT5)...")
            weights_link = "http://rostlab.org/~deepppi/embedding_repo/embedding_models/ProtTucker/ProtTucker_ProtT5.pt"
            urllib.request.urlretrieve(weights_link, str(weights_p))

        print("Loading Tucker checkpoint from: {}".format(weights_p))
        state = torch.load(weights_p)['state_dict']
        model = Tucker().to(device)
        model.load_state_dict(state)
        model.eval()

        start = time.time()
        dataset = model.single_pass(dataset)
        print("Tuckerin' took: {:.4f}[s]".format(time.time()-start))
        return dataset

    def read_inputs(self, input_p):
        # define path for storing embeddings
        emb_p = self.output_d / input_p.name.replace(".fasta", ".h5")
        if not (input_p.is_file() or emb_p.is_file()):
            print("Neither input fasta, nor embedding H5 could be found for: {}".format(input_p))
            raise FileNotFoundError

        if emb_p.is_file(): # if the embedding file already exists
            return self.read_embeddings(emb_p)
        
        elif input_p.name.endswith(".fasta"): # compute new embeddings if only FASTA available
            if self.Embedder is None: # avoid re-loading the pLM
                self.Embedder = Embedder()
            id2seq = self.read_fasta(input_p)
            id2emb = self.Embedder.get_embeddings(id2seq)
            self.Embedder.write_embeddings(emb_p, id2emb)
            keys, embeddings = zip(*id2emb.items())
            # matrix of values (protein-embeddings); n_proteins x embedding_dim
            embeddings = np.vstack(embeddings)
            return list(keys), torch.tensor(embeddings).to(device).float()

        else:
            print("The file you passed neither ended with .fasta nor .h5. " +
                  "Only those file formats are currently supported.")
            raise NotImplementedError

    def read_fasta(self, fasta_path):
        '''
            Store sequences in fasta file as dictionary with keys being fasta headers and values being sequences.
            Also, replace gap characters and insertions within the sequence as those can't be handled by ProtT5 
                when generating embeddings from sequences'.
            Also, replace special characters in the FASTA headers as those are interpreted as special tokens 
                when loading pre-computed embeddings from H5. 
        '''
        sequences = dict()
        with open(fasta_path, 'r') as fasta_f:
            for line in fasta_f:
                # get uniprot ID from header and create new entry
                if line.startswith('>'):
                    uniprot_id = line.replace('>', '').strip()
                    # replace tokens that are mis-interpreted when loading h5
                    uniprot_id = uniprot_id.replace("/", "_").replace(".", "_")
                    sequences[uniprot_id] = ''
                else:
                    # repl. all whie-space chars and join seqs spanning multiple lines
                    # drop gaps and cast to upper-case
                    sequences[uniprot_id] += ''.join(
                        line.split()).upper().replace("-", "")
        return sequences

    def read_embeddings(self, emb_p):
        start = time.time()
        h5_f = h5py.File(emb_p, 'r')
        dataset = {pdb_id: np.array(embd) for pdb_id, embd in h5_f.items()}
        keys, embeddings = zip(*dataset.items())
        # matrix of values (protein-embeddings); n_proteins x embedding_dim
        embeddings = np.vstack(embeddings)
        print("Loading embeddings from {} took: {:.4f}[s]".format(
            emb_p, time.time()-start))
        return list(keys), torch.tensor(embeddings).to(device).float()

    def read_label_mapping(self, set_ids, label_p):
        if label_p is None:
            return {set_id: None for set_id in set_ids}
        
        # in case you pass your own label mapping, you might need to adjust the function below
        with open(label_p, 'r') as in_f:
            # protein-ID : label
            label_mapping = {line.strip().split(
                ',')[0]: line.strip().split(',')[1] for line in in_f}
        return label_mapping

    def write_predictions(self, predictions):
        out_p = self.output_d / "eat_result.txt"
        with open(out_p, 'w+') as out_f:
            out_f.write(
                "Query-ID\tQuery-Label\tLookup-ID\tLookup-Label\tEmbedding distance\tNearest-Neighbor-Idx\n")
            out_f.write("\n".join(
                ["{}\t{}\t{}\t{}\t{:.4f}\t{}".format(query_id, query_label, lookup_id, lookup_label, eat_dist, nn_iter+1)
                 for query_id, query_label, lookup_id, lookup_label, eat_dist, nn_iter in predictions
                 ]))
        return None

    def pdist(self, lookup, queries, norm=2):
        return torch.cdist(lookup.unsqueeze(dim=0).double(), queries.unsqueeze(dim=0).double(), p=norm).squeeze(dim=0)

    def get_NNs(self, random=False):
        start = time.time()
        p_dist = self.pdist(self.lookup_embs, self.query_embs)
        self_hits = torch.isclose(p_dist, torch.zeros_like(p_dist), atol=1e-5)
        # replace self-hits with infinte dimension to avoid self-hit lookup
        p_dist[self_hits] = float('inf')

        if random: # this is only needed for benchmarking against random background
            print("Making RANDOM predictions!")
            nn_dists, nn_idxs = torch.topk(torch.rand_like(
                p_dist), self.num_NN, largest=False, dim=0)
        else: # infer nearest neighbor indices
            nn_dists, nn_idxs = torch.topk(
                p_dist, self.num_NN, largest=False, dim=0)

        predictions = list()
        n_test = len(self.query_ids)
        for test_idx in range(n_test):  # for all test proteins
            query_id = self.query_ids[test_idx]  # get id of test protein
            nn_idx = nn_idxs[:, test_idx]
            nn_dist = nn_dists[:, test_idx]
            for nn_iter, (nn_i, nn_d) in enumerate(zip(nn_idx, nn_dist)):
                # index of nearest neighbour (nn) in train set
                nn_i, nn_d = int(nn_i), float(nn_d)
                # get id of nn (infer annotation)
                lookup_id = self.lookup_ids[nn_i]
                lookup_label = self.lookupLabels[lookup_id]
                query_label = self.queryLabels[query_id]
                predictions.append(
                    (query_id, query_label, lookup_id, lookup_label, nn_d, nn_iter))
        end = time.time()
        print("Computing NN took: {:.4f}".format(end-start))
        return predictions


def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=(
        """
                eat.py uses Euclidean distance between embeddings to transfer annotations 
                from a lookup file to a query file.
                The input (lookup & query file) can either be passed as raw protein sequence files (*.fasta) or
                using your own, pre-computed embeddings (*.h5).
                If you only provide a FASTA file, embeddings are generated by default from ProtT5 (general purpose EAT).
                If you want to use ProtTucker(ProtT5) to transfer annotations (useful for remote structural homologs), set 'use_tucker' to 1.
                If you do not provide seperate label files linking fasta headers to annotations (optional), IDs from fasta headers of proteins in the lookup file
                are interpreted as labels. For example, if you pass a FASTA file, protein headers are transferred from 
                lookup to queries. If you pass a H5 file, keys from pre-computed embeddings are transferred.
                Providing your own labels file will usually require you to implement your own parsing function.
                By default only THE nearest neighbor is inferred. This can be changed using the --num_NN parameter.
                If you also pass labels for queries via --queryLabels, you can compute EAT performance.
            """
    ))

    # Required positional argument
    parser.add_argument('-l', '--lookup', required=True, type=str,
                        help='A path to your lookup file, stored either as fasta file (*.fasta) OR' +
                        'as pre-computed embeddings (H5-format; *.h5).')

    # Optional positional argument
    parser.add_argument('-q', '--queries', required=True, type=str,
                        help='A path to your query file, stored either as fasta file (*.fasta) OR' +
                        'as pre-computed embeddings (H5-format; *.h5).')

    # Optional positional argument
    parser.add_argument('-o', '--output', required=True, type=str,
                        help='A path to folder storing EAT results.')

    # Required positional argument
    parser.add_argument('-a', '--lookupLabels', required=False, type=str,
                        default=None,
                        help='A path to annotations for the proteins in the lookup file.' +
                        'Should be a CSV with 1st col. having protein IDs as stated in FASTA/H5 file and' +
                        '2nd col having labels.For example: P12345,Nucleus')

    # Optional positional argument
    parser.add_argument('-b', '--queryLabels', required=False, type=str,
                        default=None,
                        help='A path to annotations for the proteins in the query file. ' +
                        'Same format as --lookupLabels. Needed for EAT accuracy estimate.')

    parser.add_argument('--use_tucker', type=int,
                        default=0,
                        help="Whether to use ProtTucker(ProtT5) to generate per-protein embeddings." +
                        " Default: 0 (no tucker).")
    
    parser.add_argument('--num_NN', type=int,
                        default=1,
                        help="The number of nearest neighbors to retrieve via EAT." +
                        "Default: 1 (retrieve only THE nearest neighbor).")
    return parser


def main():
    parser = create_arg_parser()
    args = parser.parse_args()

    lookup_p = Path(args.lookup)
    query_p = Path(args.queries)
    output_d = Path(args.output)

    lookupLabels_p = None if args.lookupLabels is None else Path(
        args.lookupLabels)
    queryLabels_p = None if args.queryLabels is None else Path(
        args.queryLabels)

    num_NN = int(args.num_NN)
    assert num_NN > 0, print(
        "Only positive number of nearest neighbors can be retrieved.")

    use_tucker = int(args.use_tucker)
    use_tucker = False if use_tucker == 0 else True

    eater = EAT(lookup_p, query_p, output_d,
                use_tucker, num_NN, lookupLabels_p, queryLabels_p)
    predictions = eater.get_NNs()
    eater.write_predictions(predictions)

    if queryLabels_p is not None:
        print("Found labels to queries. Computing EAT performance ...")
        evaluator = Evaluator(predictions)
        evaluator.compute_performance()

    return None


if __name__ == '__main__':
    main()
