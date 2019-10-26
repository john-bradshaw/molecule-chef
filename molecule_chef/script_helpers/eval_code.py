
import warnings

import multiset
import numpy as np
from rdkit.Chem import QED
from rdkit import Chem


from .. import mchef_config


def evaluate_reconstruction_accuracy(padded_true_batch_first, padded_suggestions_seq_first, tb_write=None):
    total_elems = 0
    total_bags = padded_true_batch_first.shape[0]

    matched_elems = 0
    matched_bags = 0

    table_rows = ['| True  |   Predicted |   ']

    for true_elems, suggested_elems in zip(padded_true_batch_first.cpu().numpy(), padded_suggestions_seq_first.cpu().numpy().T):
        true_elems = multiset.Multiset(true_elems[true_elems != mchef_config.PAD_VALUE].tolist())
        suggested_elems = multiset.Multiset(suggested_elems[suggested_elems != mchef_config.PAD_VALUE].tolist())
        table_rows.append('| ' +
                          ' | '.join([','.join(map(str, true_elems)), ','.join(map(str, suggested_elems))]) +
                          '|   ')

        total_elems += len(true_elems)
        matched_bags += int(true_elems == suggested_elems)
        matched_elems += len(true_elems.intersection(suggested_elems))

    if tb_write is not None:
        tb_write.add_text('Reconstruction Results', '\n'.join(table_rows))

    return float(matched_bags) / total_bags, float(matched_elems) / total_elems


def get_best_qed_from_smiles_bag(text_line):
    molecules = [Chem.MolFromSmiles(mol_str) for mol_str in text_line.split('.')]
    qed_scores = []
    for mol in molecules:
        if mol is None:
            continue
        try:
            qed_scores.append(QED.qed(mol))
        except Exception as ex:
            warnings.warn("Could not find a qed (skipping): " + str(ex))
    return np.max(qed_scores) if len(qed_scores) else -np.inf
