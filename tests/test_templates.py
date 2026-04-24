import pytest
"""Tests for NotebookTemplate."""

from smallclawlm.extensions.templates import NotebookTemplate


class TestTemplates:
    """Test notebook template definitions."""

    def test_list_templates(self):
        templates = NotebookTemplate.list_templates()
        assert isinstance(templates, dict)
        assert "podcast" in templates
        assert "research_paper" in templates

    def test_get_template(self):
        template = NotebookTemplate.get_template("podcast")
        assert "description" in template
        assert "steps" in template

    def test_get_unknown_template(self):
        with pytest.raises(ValueError, match="Unknown template"):
            NotebookTemplate.get_template("nonexistent")
