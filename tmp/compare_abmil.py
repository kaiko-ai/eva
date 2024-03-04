
from eva.vision.models.networks.abmil import ABMIL
from eva.vision.models.networks.abmil_orig import ABMILClassifier


abmil = ABMIL(
    input_size=384,
    output_size=6,
    projected_input_size=128,
    hidden_size_attention=128,
    hidden_sizes_mlp=[128, 64],
    use_bias=True
)

abmil_orig = ABMILClassifier(
    input_size=384,
    output_size=6,
    projected_input_size=128,
    hidden_size_attention=128,
    hidden_sizes_mlp=[128, 64],
    use_bias=True
)
