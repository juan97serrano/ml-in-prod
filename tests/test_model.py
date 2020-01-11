"""Tests for the model module."""

from trainer import model

def test_build_model():
    """Test the build model"""
    m = model.build_model()
    assert m.layers is not None
    assert m.layers != []


