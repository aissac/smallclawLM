"""NotebookTemplate — Pre-built notebook configurations for common use cases."""


class NotebookTemplate:
    """Pre-built notebook templates for common research and generation workflows."""

    TEMPLATES = {
        "research_paper": {
            "description": "Analyze academic papers and generate summaries",
            "steps": ["add_source", "ask", "generate_report"],
        },
        "podcast": {
            "description": "Create an audio podcast from web research",
            "steps": ["research", "generate_podcast", "download"],
        },
        "study_guide": {
            "description": "Generate study materials (flashcards + quiz)",
            "steps": ["add_source", "generate_flashcards", "generate_quiz"],
        },
        "news_briefing": {
            "description": "Daily news briefing with research",
            "steps": ["research", "generate_report", "generate_mindmap"],
        },
        "code_review": {
            "description": "Analyze code repositories and generate reports",
            "steps": ["add_source", "ask", "generate_report"],
        },
    }

    @classmethod
    def list_templates(cls) -> dict:
        """List all available templates."""
        return cls.TEMPLATES

    @classmethod
    def get_template(cls, name: str) -> dict:
        """Get a specific template by name."""
        if name not in cls.TEMPLATES:
            raise ValueError(f"Unknown template: {name}. Available: {list(cls.TEMPLATES.keys())}")
        return cls.TEMPLATES[name]
