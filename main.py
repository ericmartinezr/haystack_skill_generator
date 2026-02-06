import os
from pathlib import Path
from haystack import Pipeline, tracing
from haystack_integrations.components.connectors.langfuse import LangfuseConnector
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.components.agents import Agent
from dotenv import load_dotenv

# Tools
from tools.create_skills import create_skill
from tools.read_skill import read_skill
from tools.command_runner import command_runner
from tools.find_skill import find_skill
from constants import CHAT_GENERATOR_MODEL

load_dotenv()


tracing.tracer.is_content_tracing_enabled = True


# "nemotron-3-nano"
chat_generator = OllamaChatGenerator(
    model=CHAT_GENERATOR_MODEL,
    timeout=480,
    generation_kwargs={
        "temperature": 0.1
    }
)

system_prompt = """You are a helpful AI agent expert in using and creating "Skills".

# Workflow
1. **Analyze** the user request.
2. **Find** if a relevant SKILL exists using the `find_skill` tool.
    - If a relevant SKILL exists, proceed to step 4.
    - If NO relevant SKILL exists, proceed to step 3.
3. **Create** a new SKILL using the `create_skill` tool.
    - Provide a concise query to the tool to generate the skill (e.g., "create a skill for managing docker containers").
4. **Convert** the file paths from step 3 into a list for step 5
5. **Read** the content of the SKILLs using the `read_skill` tool. Call this tool ONLY ONE TIME.
    - Use the file paths from step 4 as argument.
    - You MUST read the skills before using them.
6. **Execute** the task using the instructions and commands found in the SKILL.
    - Use the `command_runner` tool to execute commands.
    - If the skill suggests a Python script, you can write it to a file (using echo or printf command via command_runner) and run it.
    - STRICTLY FOLLOW the syntax and examples provided in the SKILL.md.

# Constraints
- Do not ask the user for clarification unless absolutely necessary. Attempt to solve it with the tools.
- If the SKILL implementation requires multiple steps, perform them.
- Always report back a brief and concise summary of the result to the user.
"""

agent = Agent(
    chat_generator=chat_generator,
    tools=[find_skill, create_skill, read_skill, command_runner],
    max_agent_steps=12,
    system_prompt=system_prompt,
    exit_conditions=["text"]
)

pipeline = Pipeline(max_runs_per_component=1)
pipeline.add_component("tracer", LangfuseConnector("Haystack Skill Generator"))
pipeline.add_component("main_agent", agent)
pipeline.draw(path=Path("pipeline.png"))


def run_agent(query: str):
    print(f"Agent running with query: {query}")
    response = pipeline.run(data={
        "main_agent": {
            "messages": [ChatMessage.from_user(query)]
        }
    })

    last_message = response["main_agent"]["messages"][-1]
    print("\nAgent Response:\n")
    print(last_message.text)


if __name__ == "__main__":
    user_query = "list the python files in the folder /home/eric/haystack-skill-generator and save them to a PDF file"
    run_agent(user_query)
