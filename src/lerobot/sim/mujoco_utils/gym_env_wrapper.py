

class GymEnvWrapper:
    pass


__gym_env_wrapper_singleton = None

def init_gym_wrapper():
    global __gym_env_wrapper_singleton
    if __gym_env_wrapper_singleton is None:
        __gym_env_wrapper_singleton = GymEnvWrapper()
    else:
        raise RuntimeError("GymEnvWrapper already initialized")

def get_gym_env_wrapper_singleton():
    global __gym_env_wrapper_singleton
    if __gym_env_wrapper_singleton is None:
        raise RuntimeError("GymEnvWrapper not initialized")
    return __gym_env_wrapper_singleton
