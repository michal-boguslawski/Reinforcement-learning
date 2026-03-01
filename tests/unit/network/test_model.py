import os
from pathlib import Path
import torch as T
from .load_model import load_model


folder_path = Path(__file__).parent.absolute()
states_path = os.path.join(folder_path, "data", "states.pt")
states = T.load(states_path)
print(states.shape)

model = load_model()

output = model(states)
features = model.backbone(states)

expected_logits = T.load(os.path.join(folder_path, "data", "expected_output_logits.pt"))
expected_values = T.load(os.path.join(folder_path, "data", "expected_output_values.pt"))
expected_features = T.load(os.path.join(folder_path, "data", "expected_features.pth"))

assert T.allclose(output.actor_logits, expected_logits, atol=1e-3)
assert T.allclose(output.critic_value, expected_values, atol=1e-3)
assert T.allclose(features.features, expected_features, atol=1e-3)
print("Test passed")
