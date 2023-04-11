def delete_model_from_state_dict(state_dict: dict):
    return {k.replace('model.', ''): v for k, v in state_dict.items()}
