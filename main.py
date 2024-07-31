"""Helper wrapper class for Pathology FMs."""

from eva.vision.models.networks.backbones._registry import BackboneModelRegistry

model = BackboneModelRegistry.load_model("kaiko_vits16")
