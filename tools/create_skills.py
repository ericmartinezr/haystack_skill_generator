from haystack import Pipeline, component
from haystack.components.agents import Agent
from haystack.tools import ComponentTool
from haystack.dataclasses import ChatMessage
from haystack.core.super_component import SuperComponent
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from haystack.dataclasses.byte_stream import ByteStream
from tools.read_example_skills import read_example_skills
from tools.write_skill import write_skill
from constants import CHAT_GENERATOR_MODEL

chat_generator = OllamaChatGenerator(
    model=CHAT_GENERATOR_MODEL,
    timeout=360,
)

agent = Agent(
    chat_generator=chat_generator,
    system_prompt="""You're a helpful AI agent expert on Skills designed by Anthropic. 
    Your job is to identify the SKILLs in the user query and create a SKILL.md file for each one.
    A SKILL is anything that can be converted to code and reused.
    
    # Workflow
    1. Identify the SKILLs in the user query
    - For example: "list files inside a folder and save them in a PDF"
    - In this example you have 2 SKILLs: "list files" and "pdf"
    2. Read the SKILLs definition and the example SKILL ONCE to understand the format with the tool `read_example_skill`.
    3. For each SKILL in step 1 write an independent SKILL.md file with the tool `write_skill`. Make them concise.
    4. STRICTLY FOLLOW the syntax and examples provided by the tool `read_example_skills`.

    # Tools available
    1. `read_example_skills`: Tool to read the example SKILL.md file
    2. `write_skill`: Tool to write a SKILL.md file
        - Parameters
            - `dir_name`: a one-word lowercase appropriate name for the directory (e.g., 'unix', 'windows', 'python', 'pdf', etc)
            - `file_content`: the content to be written

    If no commands are known to be used directly in the O.S., always fallback to Python.
    If the SKILL is correctly created (`write_skill` returns True) then return the text 'skill created', do not add anything else.
    """,
    tools=[read_example_skills, write_skill],
    exit_conditions=["text"]
)


@component
class FixedLinkContentFetcher:
    def __init__(self):
        self.fetcher = LinkContentFetcher(
            timeout=3,
            raise_on_failure=False,
            retry_attempts=2
        )

    @component.output_types(streams=list[ByteStream])
    def run(self):
        urls = ["https://agentskills.io/what-are-skills",
                "https://agentskills.io/specification"]
        return {"streams": self.fetcher.run(urls)["streams"]}


pipeline = Pipeline(max_runs_per_component=1)
pipeline.add_component("fetcher", FixedLinkContentFetcher())
pipeline.add_component("html", HTMLToDocument())
pipeline.add_component("summarizer", ChatPromptBuilder(
    template=[
        ChatMessage.from_user("""
        Consider the following definition about SKILLs (developed by Anthropic [Claude]).
        You have to summarize it and return it in clear Markdown format.
        <skill_definition>
        {% for doc in docs %}
            {{ doc.content }}
        {% endfor %}                  
        </skill_definition>
        """)],
    required_variables=["docs"]
))
pipeline.add_component("builder", ChatPromptBuilder(
    template=[
        ChatMessage.from_user("""
        <skill_definition>
        {{skill_definition}}
        </skill_definition>
                              
        <user_instructions>
        {{query}}
        </user_instructions>
        """)
    ],
    required_variables=["skill_definition", "query"]
))
pipeline.add_component("chat_summarizer", chat_generator)
pipeline.add_component("agent", agent)

pipeline.connect("fetcher.streams", "html.sources")
pipeline.connect("html.documents", "summarizer.docs")
pipeline.connect("summarizer.prompt", "chat_summarizer.messages")
pipeline.connect("chat_summarizer.replies", "builder.skill_definition")
pipeline.connect("builder.prompt", "agent.messages")


create_skill_component = SuperComponent(
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

create_skill = ComponentTool(
    name="create_skill",
    description="Use this tool to create SKILLs",
    component=create_skill_component,
    outputs_to_string={"source": "messages"},
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The user requirement improved by the main agent"
            }
        }
    }
)
