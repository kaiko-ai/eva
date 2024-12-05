"""Vision Pathology Model Backbones API."""

from eva.vision.models.networks.backbones.pathology.bioptimus import bioptimus_h_optimus_0
from eva.vision.models.networks.backbones.pathology.gigapath import prov_gigapath
from eva.vision.models.networks.backbones.pathology.histai import histai_hibou_b, histai_hibou_l
from eva.vision.models.networks.backbones.pathology.kaiko import (
    kaiko_vitb8,
    kaiko_vitb16,
    kaiko_vitl14,
    kaiko_vits8,
    kaiko_vits16,
)
from eva.vision.models.networks.backbones.pathology.lunit import lunit_vits8, lunit_vits16
from eva.vision.models.networks.backbones.pathology.mahmood import mahmood_uni
from eva.vision.models.networks.backbones.pathology.owkin import owkin_phikon, owkin_phikon_v2
from eva.vision.models.networks.backbones.pathology.paige import paige_virchow2

__all__ = [
    "kaiko_vitb16",
    "kaiko_vitb8",
    "kaiko_vitl14",
    "kaiko_vits16",
    "kaiko_vits8",
    "owkin_phikon",
    "owkin_phikon_v2",
    "lunit_vits16",
    "lunit_vits8",
    "mahmood_uni",
    "bioptimus_h_optimus_0",
    "prov_gigapath",
    "histai_hibou_b",
    "histai_hibou_l",
    "paige_virchow2",
]
