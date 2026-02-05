from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.tools import ComponentTool
from haystack.dataclasses import ChatMessage
from haystack.core.super_component import SuperComponent
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from tools.read_skills import read_skills

chat_generator = OllamaChatGenerator(
    model="kimi-k2.5:cloud",
    timeout=360,
)

agent = Agent(
    chat_generator=chat_generator,
    system_prompt="""
    You're a helpful AI agent expert on the SKILL definition developed by Anthropic.
    Read the name and description returned by the tool `read_skills` and find the appropriate SKILL
    for the user query.
    If you find a SKILL return the data, otherwise return 'skill not found'.
    """,
    tools=[read_skills],
    exit_conditions=["text"]
)

agent.warm_up()


pipeline = Pipeline()
pipeline.add_component("builder", ChatPromptBuilder(
    template=[
        ChatMessage.from_user("""
        <user_instructions>
        {{query}}
        </user_instructions>
        """)
    ],
    required_variables=["query"]
))
pipeline.add_component("agent", agent)
pipeline.connect("builder.prompt", "agent.messages")


find_skill_component = SuperComponent(
    pipeline=pipeline,
    input_mapping={
        # Mapea la variable "query" del componente "builder" a la variable "query"
        "query": ["builder.query"]
    },
    output_mapping={
        # Mapea el output "messages" del componente "agent" a "messages"
        "agent.messages": "messages"
    }
)

find_skill = ComponentTool(
    name="find_skill",
    description="Use this tool to find SKILLs",
    component=find_skill_component,
    outputs_to_string={"source": "messages"},
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user requirement"
            }
        }
    }
)
