import re
from typing import Dict, Any, List

def apply_diff(current_model: Dict[str, Any], diff_ops: List[str]) -> Dict[str, Any]:
    """
    Applies a list of MFR diff operations to the current STM model.
    
    Supported Ops:
    - ADD_CONSTRAINT: <content>
    - MODIFY_ACTION: <content>
    - DROP_ENTITY: <content>
    
    Args:
        current_model: The current STM model dictionary.
        diff_ops: List of string operations (e.g. ["ADD_CONSTRAINT: budget < 5000"]).
        
    Returns:
        The updated model dictionary (new copy).
    """
    # Create a deep copy to avoid mutating the original state input directly if needed,
    # though for simple dicts in this context shallow copy + field init is often enough.
    # We'll just ensure the specific fields exist.
    new_model = current_model.copy()
    
    if "constraints" not in new_model:
        new_model["constraints"] = []
    if "actions" not in new_model:
        new_model["actions"] = [] # or dict, depending on schema. Let's assume list of strings for now.
    if "entities" not in new_model:
        new_model["entities"] = {} # or list. Let's assume dict for structured entities or list for simple names.
        
    # Standardize schema for safety
    if not isinstance(new_model["constraints"], list):
        new_model["constraints"] = []
    
    # We'll assume 'entities' is a list of strings for simple entity tracking, 
    # or a dict if we want values. Based on the user request "DROP_ENTITY", 
    # let's assume it's a list or dict keys. Let's support List[str] for simplicity first 
    # as per the examples in the prompt, or Dict if we want values.
    # MFR prompt examples suggest: "Entity: List_of_Products".
    # Let's use a List[str] for now effectively.
    if not isinstance(new_model.get("entities"), list):
         # If it's a dict, we might just drop keys. 
         pass 

    for op in diff_ops:
        try:
            op_type, content = op.split(":", 1)
            op_type = op_type.strip().upper()
            content = content.strip()
            
            # Remove quotes if present
            if content.startswith('"') and content.endswith('"'):
                content = content[1:-1]
            if content.startswith("'") and content.endswith("'"):
                content = content[1:-1]

            if op_type == "ADD_CONSTRAINT":
                if content not in new_model["constraints"]:
                    new_model["constraints"].append(content)
            
            elif op_type == "MODIFY_ACTION":
                # For actions, we might append or replace. 
                # Simple implementation: Append as a new allowed action/rule.
                if "actions" not in new_model:
                    new_model["actions"] = []
                if isinstance(new_model["actions"], list):
                    if content not in new_model["actions"]:
                        new_model["actions"].append(content)
            
            elif op_type == "DROP_ENTITY":
                # Remove from entities list if present
                if isinstance(new_model.get("entities"), list):
                    new_model["entities"] = [e for e in new_model["entities"] if e != content]
                elif isinstance(new_model.get("entities"), dict):
                     if content in new_model["entities"]:
                         del new_model["entities"][content]

        except ValueError:
            # Malformed op string (missing colon etc)
            continue
            
    return new_model
