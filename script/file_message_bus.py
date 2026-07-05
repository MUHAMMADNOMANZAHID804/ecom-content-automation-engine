import json
import tempfile
from typing import Any, Union

def maybe_offload(*args) -> Union[str, Any]:
    """
    Checks if the data exceeds 2,000 characters. 
    If it does, writes it to a temporary file and returns a string prefixed with 'FILE_PATH:'.
    If it is below the threshold, returns the original data.
    Handles both maybe_offload(data) and maybe_offload(label, data) signatures.
    """
    if len(args) == 2:
        label, data = args
    elif len(args) == 1:
        label, data = "data", args[0]
    else:
        return ""

    try:
        # Serialize to check length
        serialized = json.dumps(data)
    except TypeError:
        serialized = str(data)
        
    if len(serialized) > 2000:
        tf = tempfile.NamedTemporaryFile(delete=False, mode='w', prefix=f"{label}_", suffix=".json")
        tf.write(serialized)
        tf.close()
        return f"FILE_PATH:{tf.name}"
        
    return data
