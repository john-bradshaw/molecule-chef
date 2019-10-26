
import numpy as np

from molecule_chef.chem_ops import rdkit_general_ops
from molecule_chef.chem_ops import rdkit_featurization_ops


def test_mol_to_atom_feats_and_adjacency_list():
    smiles_to_test = "[CH3:1][C:2](=[O:3])[NH:4][C:5]1=[CH:6][CH:7]=[C:8]([OH:9])[CH:10]=[CH:11]1"

    mol = rdkit_general_ops.get_molecule(smiles_to_test, True)
    am_to_idx_map = rdkit_general_ops.create_atom_map_indcs_map(mol)

    feats = rdkit_featurization_ops.mol_to_atom_feats_and_adjacency_list(mol, am_to_idx_map)

    # Check atom map to idx map is the same
    assert feats.atom_map_to_idx_map == am_to_idx_map
    idx_to_am_map = {v:k for k,v in am_to_idx_map.items()}

    # Check the bonds all registered correctly
    def convert_list_into_set_of_am_fsets(list_in):
        tuple_converter = lambda x: frozenset([idx_to_am_map[a] for a in x])
        return set(map(tuple_converter, list_in))

    expected_single_bonds = set([frozenset(x) for x in [{1,2}, {2,4}, {4,5}, {11,5}, {6,7}, {10,8}, {8,9}]])
    assert expected_single_bonds == convert_list_into_set_of_am_fsets(feats.adj_lists_for_each_bond_type['single'])

    expected_double_bonds = set([frozenset(x) for x in [(2,3), (5,6), (7,8), (11,10)]])
    assert expected_double_bonds == convert_list_into_set_of_am_fsets(feats.adj_lists_for_each_bond_type['double'])

    assert len(feats.adj_lists_for_each_bond_type['triple']) == 0

    # Now we just test some  parts of the atom features
    #  am 3 is oxygen.
    am3_idx = am_to_idx_map[3]
    expect_atom_oh_part = np.zeros(72)
    expect_atom_oh_part[42] = 1
    np.testing.assert_array_equal(feats.atom_feats[am3_idx, :72], expect_atom_oh_part)

    assert feats.atom_feats[am3_idx, -1] == 0., "this part should not be aromatic"
    assert feats.atom_feats[am3_idx, -2] == 8., "atomic number of oxygen is 16"
    assert feats.atom_feats[am3_idx, -4] == 0., "no Hs attached to this oxygen"

    #  am 9 is oxygen off the ring.
    am9_idx = am_to_idx_map[9]
    assert feats.atom_feats[am9_idx, -4] == 1., "should be one H on this oxygen"

