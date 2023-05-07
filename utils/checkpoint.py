def delete_model_from_state_dict(state_dict: dict):
    return {startswith_delete(k, 'model.'): v for k, v in state_dict.items()}

def startswith_delete(key, start_str):
    if key.startswith(start_str):
        return key[len(start_str):]
    return key
