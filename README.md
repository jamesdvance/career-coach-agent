# Career Coach Agent

A simple agentic workflow to help job seekers prioritize their skill-building based on the jobs they most admire.

Built with [LangGraph](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/) and prioritizes simplicity and speed of deployment.

## Environment Variables

The following environment variables are required:

- **`ANTHROPIC_API_KEY`** - API key for Claude (default model provider)
- **`OPENAI_API_KEY`** - API key for OpenAI GPT models (alternative provider)

Set at least one of these API keys depending on which model vendor you want to use.