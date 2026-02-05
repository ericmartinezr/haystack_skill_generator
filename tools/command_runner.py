import subprocess
import shlex
from haystack import Pipeline
from haystack.tools import ComponentTool, tool
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.core.super_component import SuperComponent
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder

WHITELIST_COMMANDS = [
    "ls",
    "cat",
    "pwd",
    "mkdir",
    "touch",
    "echo",
    "printf",
    "python",
    "find"
]

# TODO: Add blacklisted commands, basically the opposite of WHITELIST_COMMANDS
BLACKLIST_COMMANDS = []

BLACKLIST_FOLDERS = [".venv", "node_modules/", "__pycache__"]


# Algunos comandos validos para shell
SHELL_SYNTAX = [
    "&&", "||", ";", ";;", "&",
    "|", "|&",
    ">", ">>", "<",
    "<<", "<<-", "<<<",
    "2>", "2>>", "&>", ">&", "<&"
]


@tool
def run_command(command: str) -> dict:
    try:
        args = shlex.split(command)
        if args[0] not in WHITELIST_COMMANDS:
            raise ValueError(f"The command {args[0]} is not allowed.")

        # Si alguno de los parametros existe en SHELL_SYNTAX, significa que tiene que pasar por shell
        is_shell = False
        if any(True if arg in SHELL_SYNTAX else False for arg in args):
            is_shell = True

        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            check=True,
            shell=is_shell
        )

        return {
            "result": result.stdout,
            "error": result.stderr
        }
    except subprocess.CalledProcessError as e:
        return {
            "result": e.stderr,
            "error": str(e)
        }
    except Exception as e:
        return {
            "result": "",
            "error": str(e)
        }


chat_generator = OllamaChatGenerator(
    model="kimi-k2.5:cloud",
    timeout=360,
)

agent = Agent(
    chat_generator=chat_generator,
    system_prompt=f"""
    You're a helpful AI agent expert on command line programming.
    You'll receive a command to execute, it can be either a Python script or an OS command.
    Review the command and you will execute it with the tool `run_command` ONLY if it's safe, 
    otherwise notify the user you can't execute it.

    # Constraints
    - Only use the following commands: {WHITELIST_COMMANDS}. Modify the original command if possible.
    - The command MUST ignore the following folders: {BLACKLIST_FOLDERS}.
    """,
    tools=[run_command],
    raise_on_tool_invocation_failure=True,
    exit_conditions=["text"]
)

agent.warm_up()


pipeline = Pipeline(max_runs_per_component=1)
pipeline.add_component("builder", ChatPromptBuilder(
    template=[
        ChatMessage.from_user("<command>{{command}}</command>")
    ],
    required_variables=["command"]
))
pipeline.add_component("agent", agent)
pipeline.connect("builder.prompt", "agent.messages")

command_runner_component = SuperComponent(
    pipeline=pipeline,
    input_mapping={
        # Mapea la variable "command" del component "builder" a la variable "command"
        "command": ["builder.command"]
    },
    output_mapping={
        # Mapea el output "messages" del componente "agent" a "messages"
        "agent.messages": "messages"
    }
)

command_runner = ComponentTool(
    component=command_runner_component,
    name="command_runner",
    description="Use this tool to run command line commands",
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command to be executed"
            }
        }
    }
)
