import os

# Keep unit tests offline even if a developer .env enables tracing or contains keys.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_API_KEY"] = ""
os.environ["LANGSMITH_PROJECT"] = "mlops-chatbot-demo"
os.environ["OPENAI_API_KEY"] = ""
os.environ["ALLOW_DEMO_FAULTS"] = "false"
