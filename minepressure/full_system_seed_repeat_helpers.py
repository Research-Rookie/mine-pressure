"""Paper algorithm components; no experiment execution or stored observations."""
import torch
from .conditional_displacement_operator import ConditionalDisplacementOperator
from .family_conditioned_operator import FamilyConditionedOperator
from .full_train_extreme_operator import FullTrainWideFaceOperator

def numeric_initialization(statistics, seed):
    torch.manual_seed(seed)
    original = ConditionalDisplacementOperator(statistics).state_dict()
    torch.manual_seed(seed)
    family = FamilyConditionedOperator(statistics, 'shared_wide')
    missing = family.load_state_dict({k: v for k, v in original.items() if not k.startswith(('local.', 'read.'))}, strict=False)
    assert not missing.unexpected_keys
    assert all((k.startswith('decoders.') or k == 'family_index' for k in missing.missing_keys))
    torch.manual_seed(seed)
    model = FullTrainWideFaceOperator(statistics)
    missing = model.load_state_dict({k: v for k, v in family.state_dict().items() if not k.startswith('decoders.')}, strict=False)
    assert not missing.unexpected_keys and all((k.startswith('decoders.') for k in missing.missing_keys))
    return model
