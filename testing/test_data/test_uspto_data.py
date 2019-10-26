
import multiset

from molecule_chef.data import uspto_data


def test_actionset_from_uspto_line():
    change_str = "11-14;11-13"
    expected = {11,13,14}
    actual = uspto_data.actionset_from_uspto_line(change_str)
    assert actual == expected


def test_trsfm_to_reactant_product_multisets():
    reactants = "[CH3:15][C:16](=[O:17])[CH3:18].[CH3:3][O:4][c:5]1[cH:6][cH:7][c:8]([S:11](=[O:12])(=[O:13])[Cl:14])[cH:9][cH:10]1.[I-:1].[Na+:2]"
    products = "[CH3:3][O:4][c:5]1[cH:6][cH:7][c:8]([S:11](=[O:12])[O-:13])[cH:9][cH:10]1.[Na+:2]"
    bond_changes = "11-14;11-13"

    expected_reactants = multiset.FrozenMultiset(['COc1ccc(S(=O)(=O)Cl)cc1'])
    expected_products = multiset.FrozenMultiset(['COc1ccc(S(=O)[O-])cc1'])

    canconical_reactants_set, canonical_product_set = uspto_data.trsfm_to_reactant_product_multisets(reactants, products, bond_changes)

    assert canconical_reactants_set == expected_reactants
    assert canonical_product_set == expected_products
