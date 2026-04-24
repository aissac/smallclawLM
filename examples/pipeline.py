"""Pipeline example."""

from smallclawlm import Pipeline

# Create a pipeline: add source → research → generate → ask
result = (
    Pipeline()
    .add_source("https://arxiv.org/abs/2401.12345")
    .research("quantum error correction advances in 2025")
    .ask("What are the top 3 breakthroughs?")
    .execute()
)

print(result.outputs)
