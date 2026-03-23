from .registry import WRAPPERS


def prepare_wrappers(config: dict | None) -> list:
    wrappers = []
    if config:
        for wrapper_name, wrapper_kwargs in config.items():
            kwargs = wrapper_kwargs.copy()
            wrapper = WRAPPERS.get(wrapper_name, None)
            if wrapper_name == "resize_observation" and isinstance(kwargs["shape"], int):
                kwargs["shape"] = (kwargs["shape"], kwargs["shape"])
            if wrapper is None:
                raise ValueError(f"Unknown wrapper '{wrapper_name}'")
            wrappers.append(lambda env, w=wrapper, kw=kwargs: w(env=env, **kw))
    return wrappers
