
import typing
import collections

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

_electroneg = {'Ag': 1.93, 'Al': 1.61, 'Ar': 3.98,
              'As': 2.18, 'Au': 2.54, 'B':  2.04,
              'Ba': 0.89, 'Be': 1.57, 'Bi': 2.02,
              'Br': 2.96, 'C':  2.55, 'Ca': 1.0,
              'Cd': 1.69, 'Ce': 1.12, 'Cl': 3.16,
              'Co': 1.88, 'Cr': 1.66, 'Cs': 0.79,
              'Cu': 1.90, 'Dy': 1.22, 'Eu': 3.98,
              'F':  3.98, 'Fe': 1.83, 'Ga': 1.81,
              'Ge': 2.01, 'H':  2.20, 'He': 3.98,
              'Hf': 1.3,  'Hg': 2.0,  'I':  2.66,
              'In': 1.78, 'Ir': 2.20, 'K':  0.82,
              'La': 1.10, 'Li': 0.98, 'Mg': 1.31,
              'Mn': 1.55, 'Mo': 2.16, 'N':  3.04,
              'Na': 0.93, 'Nd': 1.14, 'Ni': 1.91,
              'O':  3.44, 'Os': 2.20, 'P':  2.19,
              'Pb': 2.33, 'Pd': 2.20, 'Pr': 1.13,
              'Pt': 2.28, 'Pu': 1.28, 'Ra': 0.9,
              'Rb': 0.82, 'Re': 1.9,  'Rh': 2.28,
              'Rn': 3.98, 'Ru': 2.2,  'S':  2.58,
              'Sb': 2.05, 'Sc': 1.36, 'Se': 2.55,
              'Si': 1.90, 'Sm': 1.17, 'Sn': 1.96,
              'Sr': 0.95, 'Ta': 1.5,  'Tb': 3.98,
              'Tc': 1.9,  'Te': 2.1,  'Th': 1.3,
              'Ti': 1.54, 'Tl': 1.62, 'Tm': 1.25,
              'U':  1.38, 'V':  1.63, 'W':  2.36,
              'Xe': 2.6,  'Y':  1.22, 'Yb': 3.98,
              'Zn': 1.65, 'Zr': 1.33}

class MolAsAdjList:
    def __init__(self, atom_feats: np.ndarray,
                 adj_lists_for_each_bond_type: typing.Mapping[str, typing.List[typing.Tuple[int, int]]],
                 atom_map_to_idx_map):
        self.atom_feats = atom_feats
        self.adj_lists_for_each_bond_type = adj_lists_for_each_bond_type
        self.atom_map_to_idx_map = atom_map_to_idx_map


class AtomFeatParams:
    def __init__(self):
        self.atom_types = ['Ag', 'Al', 'Ar', 'As', 'Au', 'B', 'Ba', 'Be', 'Bi', 'Br', 'C',
                    'Ca', 'Cd', 'Ce', 'Cl', 'Co', 'Cr', 'Cs', 'Cu', 'Dy', 'Eu', 'F',
                    'Fe', 'Ga', 'Ge', 'H', 'He', 'Hf', 'Hg', 'I', 'In', 'Ir', 'K', 'La',
                    'Li', 'Mg', 'Mn', 'Mo', 'N', 'Na', 'Nd', 'Ni', 'O', 'Os', 'P', 'Pb',
                    'Pd', 'Pr', 'Pt', 'Rb', 'Re', 'Rh', 'Ru', 'S', 'Sb', 'Sc', 'Se',
                    'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Te', 'Ti', 'Tl', 'V', 'W', 'Xe', 'Y',
                    'Yb', 'Zn', 'Zr']
        self.degrees = [0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.,  10.]
        self.explicit_valences = [0., 1., 2., 3., 4., 5., 6., 7., 8., 10., 12., 14.]
        self.atom_feature_length = len(self.atom_types) + len(self.degrees) + len(self.explicit_valences) + 8
        self.bond_names = ['single', 'double', 'triple']
        self.num_bond_types = len(self.bond_names)

    def get_bond_name(self, bond):
        bt = bond.GetBondType()
        return {
            Chem.rdchem.BondType.SINGLE: 'single',
            Chem.rdchem.BondType.DOUBLE: 'double',
            Chem.rdchem.BondType.TRIPLE: 'triple',
        }[bt]


def mol_to_atom_feats_and_adjacency_list(mol: AllChem.Mol, atom_map_to_index_map=None,
                                         params: AtomFeatParams=None) -> MolAsAdjList:
    params = AtomFeatParams() if params is None else params
    atoms = mol.GetAtoms()
    num_atoms = len(atoms)

    node_feats = np.zeros((num_atoms, params.atom_feature_length), dtype=np.float32)
    idx_to_atom_map = np.zeros(num_atoms, dtype=np.float32)

    if atom_map_to_index_map is None:
        # then we will create this map
        atom_map_to_index_map = {}
        use_supplied_idx_flg = False
    else:
        # we will use the mapping given
        use_supplied_idx_flg = True
        assert set(atom_map_to_index_map.values()) == set(range(len(atoms))), \
            "if give pre supplied ordering it must be the same size as the molecules trying to order"

    # First we will create the atom features and the mappings
    for atom in atoms:
        props = atom.GetPropsAsDict()
        am = props['molAtomMapNumber']  # the atom mapping in the file
        if use_supplied_idx_flg:
            idx = atom_map_to_index_map[am]
        else:
            idx = atom.GetIdx()  # goes from 0 to A-1
            atom_map_to_index_map[am] = idx
        idx_to_atom_map[idx] = am
        atom_features = get_atom_features(atom, params)
        node_feats[idx, :] = atom_features

    # Now we will go through and create the adjacency lists
    adjacency_lists = {k:[] for k in params.bond_names}
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        props_b = begin.GetPropsAsDict()
        props_e = end.GetPropsAsDict()
        am_b = props_b['molAtomMapNumber']
        am_e = props_e['molAtomMapNumber']
        ix_b = atom_map_to_index_map[am_b]
        ix_e = atom_map_to_index_map[am_e]

        bond_name = params.get_bond_name(bond)
        adjacency_lists[bond_name].append((ix_b, ix_e))

    # Finally we pack all the results together
    results = MolAsAdjList(node_feats, adjacency_lists, atom_map_to_index_map)
    return results


def get_atom_features(atom, params: AtomFeatParams):
    # the number of features and their Indices are shown in comments, although be cautious as these
    # may change as we decide what features to give.
    return np.array(onek_encoding_unk(atom.GetSymbol(), params.atom_types, False)  # 72 [0-71]
                    + onek_encoding_unk(atom.GetDegree(), params.degrees, False)  # 9  [72-80]
                    + onek_encoding_unk(atom.GetExplicitValence(),
                                                         params.explicit_valences, False)  # 12 [81-92]
                    + onek_encoding_unk(atom.GetHybridization(),
                                                         [Chem.rdchem.HybridizationType.SP,
                                                          Chem.rdchem.HybridizationType.SP2,
                                                          Chem.rdchem.HybridizationType.SP3, 0], True)  # 4 [93-96]
                    + [atom.GetTotalNumHs()]  # 1 [97]
                    + [_electroneg[atom.GetSymbol()]]  # 1 [98]
                    + [atom.GetAtomicNum()]  # 1 [99]
                    + [atom.GetIsAromatic()], dtype=np.float32)  # 1 [100]

def get_bond_feats(bond):
    bt = bond.GetBondType()
    return np.array(
        [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE],
        dtype=np.float32)

def onek_encoding_unk(x, allowable_set, if_missing_set_as_last_flag):
    if x not in set(allowable_set):
        if if_missing_set_as_last_flag:
            x = allowable_set[-1]
        else:
            raise RuntimeError
    return list(map(lambda s: x == s, allowable_set))

