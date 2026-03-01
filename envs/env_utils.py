from .registry import WRAPPERS


def prepare_wrappers(config: dict | None) -> list:
    wrappers = []
    if config:
        for wrapper_name, wrapper_kwargs in config.items():
            wrapper = WRAPPERS.get(wrapper_name, None)
            if wrapper is None:
                raise ValueError(f"Unknown wrapper '{wrapper_name}'")
            wrappers.append(lambda env, w=wrapper, kw=wrapper_kwargs: w(env=env, **kw))
    return wrappers
