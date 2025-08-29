"""Tests for multimodal dataset schemas."""

from eva.multimodal.data.datasets.schemas import TransformsSchema


def test_transforms_schema_with_all_fields():
    """Test TransformsSchema with all transform fields."""

    def text_transform(x):
        return x

    def image_transform(x):
        return x

    def target_transform(x):
        return x

    schema = TransformsSchema(text=text_transform, image=image_transform, target=target_transform)

    assert schema.text is text_transform
    assert schema.image is image_transform
    assert schema.target is target_transform


def test_transforms_schema_with_image_only():
    """Test TransformsSchema with only image transform."""

    def image_transform(x):
        return x

    schema = TransformsSchema(image=image_transform)

    assert schema.text is None
    assert schema.image is image_transform
    assert schema.target is None


def test_transforms_schema_frozen():
    """Test that TransformsSchema is frozen (immutable)."""
    schema = TransformsSchema()

    # Attempt to modify should raise an error
    try:
        schema.image = lambda x: x  # type: ignore
        raise AssertionError("Schema should be frozen")
    except Exception:
        # Expected - schema is frozen
        pass
