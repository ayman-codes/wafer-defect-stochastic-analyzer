import pytest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from proxel import Proxel, ProxelTree

def test_proxel_sorting_order():
    """Asserts deterministic sorting of Proxels by state and age vectors."""
    p1 = Proxel(state=1, age_vector=(1.0, 2.0), probability=0.5)
    p2 = Proxel(state=1, age_vector=(0.0, 5.0), probability=0.5)
    p3 = Proxel(state=2, age_vector=(0.0, 0.0), probability=0.5)
    
    proxel_list = [p1, p2, p3]
    proxel_list.sort()
    
    assert proxel_list[0] == p2
    assert proxel_list[1] == p1
    assert proxel_list[2] == p3

def test_proxeltree_pruning():
    """Validates mathematical isolation of pruning limits."""
    tree = ProxelTree()
    tree.add(Proxel(state=1, age_vector=(0.0, 0.0), probability=1e-8))
    tree.add(Proxel(state=1, age_vector=(1.0, 1.0), probability=0.5))
    
    assert len(tree) == 2
    tree.prune(threshold=1e-7)
    
    assert len(tree) == 1
    assert tree.proxels[0].probability == 0.5
    assert tree.accumulated_loss == 1e-8

def test_proxeltree_merging():
    """Ensures deterministic weighted mass combinations for close age vectors."""
    tree = ProxelTree()
    p1 = Proxel(state=1, age_vector=(1.0, 1.0), probability=0.6, rewards=10.0)
    p2 = Proxel(state=1, age_vector=(1.05, 1.05), probability=0.4, rewards=5.0)
    
    tree.batch_add([p1, p2])
    tree.merge(age_tolerance=0.1)
    
    assert len(tree) == 1
    merged = tree.proxels[0]
    
    # Assert mass conservation
    assert merged.probability == 1.0
    
    # Assert weighted average ages: (1.0*0.6 + 1.05*0.4) / 1.0 = 1.02
    assert merged.age_vector[0] == pytest.approx(1.02)
    assert merged.age_vector[1] == pytest.approx(1.02)
    
    # Assert weighted rewards: (10.0*0.6 + 5.0*0.4) / 1.0 = 8.0
    assert merged.rewards == pytest.approx(8.0)