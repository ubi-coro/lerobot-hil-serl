def get_wrapper_callable(env, attr_name):
    """
    Recursively search through Gym wrappers for an attribute (property or method).
    Returns a callable that returns the current value of that attribute.
    """
    if hasattr(env, attr_name):
        attr = getattr(env, attr_name)
        if callable(attr):
            return attr  # already a method
        else:
            return lambda: env.attr  # wrap property or static value as callable
    elif hasattr(env, "env"):
        return get_wrapper_callable(env.env, attr_name)
    else:
        raise AttributeError(f"Attribute '{attr_name}' not found in any wrapper.")