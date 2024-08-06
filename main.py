"""Helper wrapper class for Pathology FMs."""

from eva.vision.models import backbones

# model = backbones.BackboneModelRegistry.load_model("pathology/kaiko_vits16")
model = backbones.BackboneModelRegistry.load_model("pathology/owkin_phikon")
print(type(model))
