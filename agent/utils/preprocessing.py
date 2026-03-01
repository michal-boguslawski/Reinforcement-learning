import torch as T

from models.models import ActionSpaceType, Observation


def preprocess_batch(batch: Observation, action_space_type: ActionSpaceType) -> Observation:
    state = T.as_tensor(batch.state, dtype=T.float32)
    logits = T.as_tensor(batch.logits, dtype=T.float32)
    
    action = T.as_tensor(
        batch.action,
        dtype=T.int64 if action_space_type == "discrete" else T.float32
    )
    if action_space_type == "discrete":
        action = action.unsqueeze(-1)
    reward = T.as_tensor(batch.reward, dtype=T.float32)
    done = T.as_tensor(batch.done, dtype=T.float32)
    value = T.as_tensor(batch.value, dtype=T.float32) if batch.value is not None else None
    log_probs = T.as_tensor(batch.log_probs, dtype=T.float32)
    core_state = T.as_tensor(batch.core_state, dtype=T.float32) if batch.core_state is not None else None
    preprocessed_batch = type(batch)(
        state=state,
        logits=logits,
        action=action,
        reward=reward,
        done=done,
        value=value,
        log_probs=log_probs,
        core_state=core_state,
    )
    return preprocessed_batch
