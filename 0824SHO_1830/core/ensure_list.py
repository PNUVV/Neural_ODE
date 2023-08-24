def ensure_list(*values):
    max_length = max(len(value) if isinstance(value, list) else 1 for value in values)
    return [[value] * max_length if not isinstance(value, list) else value * (max_length // len(value)) for value in values]
