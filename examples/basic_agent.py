"""Basic SmallClawLM agent example."""

from smallclawlm import NLMAgent

# Create agent with default NLMModel (zero external tokens!)
agent = NLMAgent()

# Run a research + generation task
result = agent.run(
    "Research the latest advances in fusion energy and create a podcast about it"
)

print(result)
