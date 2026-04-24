# Tool Catalog — Available Actions for SmallClawLM

All tools listed below can be called as Python functions inside code blocks.

## Brain Routing Tools

### ask_notebook(question, notebook_id)
Ask a specialist notebook a question. Returns a grounded answer from that notebook's sources.
- **question** (str): The question to ask. Be specific.
- **notebook_id** (str, optional): Which specialist brain to query. If omitted, uses the default brain.
- Example:
  ```python
  ask_notebook(question="What are the latest breakthroughs in fusion energy?")
  ```
- Example with specific brain:
  ```python
  ask_notebook(question="Summarize the key findings", notebook_id="research-brain-abc123")
  ```

## Research Tools

### deep_research(query, mode)
Run deep web research on a topic. Returns a comprehensive report.
- **query** (str): Research question or topic.
- **mode** (str): Either "fast" or "deep" (default: "deep").
- Example:
  ```python
  deep_research(query="fusion energy breakthroughs 2024", mode="deep")
  ```

## Artifact Generation Tools

### generate_podcast(instructions)
Generate an audio overview podcast from notebook sources.
- **instructions** (str, optional): Custom podcast instructions.
- Example:
  ```python
  generate_podcast(instructions="Focus on recent breakthroughs, casual tone")
  ```

### generate_video(style)
Generate a video explainer from notebook sources.
- **style** (str): "whiteboard" or "animated" (default: "whiteboard").
- Example:
  ```python
  generate_video(style="animated")
  ```

### generate_quiz(difficulty)
Generate a quiz from notebook sources.
- **difficulty** (str): "easy", "medium", or "hard" (default: "medium").
- Example:
  ```python
  generate_quiz(difficulty="hard")
  ```

### generate_mind_map()
Generate a visual mind map from notebook sources. No parameters.
- Example:
  ```python
  generate_mind_map()
  ```

### generate_report()
Generate a structured report from notebook sources. No parameters.
- Example:
  ```python
  generate_report()
  ```

## Source Management Tools

### add_source(url, title)
Add a source (URL, YouTube, PDF) to the current notebook.
- **url** (str): URL of the source to add.
- **title** (str, optional): Display title for the source.
- Example:
  ```python
  add_source(url="https://arxiv.org/abs/2401.12345", title="Fusion Research Paper")
  ```

### list_sources()
List all sources in the current notebook. No parameters.
- Example:
  ```python
  list_sources()
  ```

### create_notebook(title)
Create a new NotebookLM notebook.
- **title** (str): Title for the new notebook.
- Example:
  ```python
  create_notebook(title="Fusion Energy Research")
  ```

## Direct Response

### direct_response(content)
Respond directly to the user without using any tool. Use this when you have enough information to answer.
- **content** (str): Your response to the user.
- Example:
  ```python
  direct_response(content="Based on my research, fusion energy has made significant progress in three key areas...")
  ```
