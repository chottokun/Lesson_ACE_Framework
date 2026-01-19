import importlib.util
import sys
import os

# Bypass package import to avoid 'sentence_transformers' dependency for this unit test
file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/ace_rm/utils/stm_manager.py'))
spec = importlib.util.spec_from_file_location("stm_manager", file_path)
stm_manager = importlib.util.module_from_spec(spec)
sys.modules["stm_manager"] = stm_manager
spec.loader.exec_module(stm_manager)

apply_diff = stm_manager.apply_diff

def test_apply_diff():
    # Initial Model
    initial_model = {
        "constraints": ["Time < 10:00"],
        "actions": ["greet"],
        "entities": ["User"]
    }

    # Op List
    ops = [
        "ADD_CONSTRAINT: Budget <= 5000",
        "MODIFY_ACTION: use_card_key",
        "DROP_ENTITY: User"
    ]
    
    # Expected Result
    # constraints -> ["Time < 10:00", "Budget <= 5000"]
    # actions -> ["greet", "use_card_key"]
    # entities -> [] (if dict logic assumed list for "User") or handling.
    # Logic in code: if list, filter. if dict, del key.
    
    new_model = apply_diff(initial_model, ops)
    
    print("Initial:", initial_model)
    print("Diff Ops:", ops)
    print("New Model:", new_model)
    
    assert "Budget <= 5000" in new_model["constraints"]
    assert "use_card_key" in new_model["actions"]
    assert "User" not in new_model["entities"]
    
    print("Test Passed!")

if __name__ == "__main__":
    test_apply_diff()
