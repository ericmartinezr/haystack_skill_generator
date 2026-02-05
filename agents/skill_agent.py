from haystack import Pipeline, super_component, component
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.components.fetchers.link_content import LinkContentFetcher
from haystack.components.converters import HTMLToDocument
from haystack.components.builders.chat_prompt_builder import ChatPromptBuilder
from tools.read_example_skills import read_example_skills
from tools.write_skill import write_skill


# Este???
# https://haystack.deepset.ai/tutorials/43_building_a_tool_calling_agent

# https://haystack.deepset.ai/tutorials/44_creating_custom_supercomponents#understanding-the-super_component-decorator
@component
class SkillAgent:

    def __init__(self):
        self.pipeline = Pipeline()
        self.pipeline.add_component("fetcher", LinkContentFetcher())
        self.pipeline.add_component("html", HTMLToDocument())
        self.pipeline.add_component("builder", ChatPromptBuilder(
            template=[
                ChatMessage.from_user("""
                Based on the following definition of SKILL follow the user instructions:

                <skill_definition>
                {% for doc in docs %}
                {{ doc.content }}
                {% endfor %}
                </skill_definition>

                <user_instructions>
                {{skill_prompt}}
                </user_instructions>
                """)
            ],
            required_variables=["docs", "skill_prompt"]
        ))
        self.pipeline.add_component("agent", self._agent())
        self.pipeline.connect("fetcher.streams", "html.sources")
        self.pipeline.connect("html.documents", "builder.docs")
        self.pipeline.connect("builder", "agent")

    def run(self, skill_prompt: str):
        result = self.pipeline.run(data={
            "fetcher": {
                "urls": [
                    "https://agentskills.io/what-are-skills",
                    "https://agentskills.io/specification"]
            },
            "builder": {
                "user_prompt": skill_prompt
            },
            "tracer": {
                "invocation_context": {
                    "test": "agent_with_tools"
                }
            }
        })
        return result["agent"]["last_message"].text

    def _agent(self):
        chat_generator = OllamaChatGenerator(
            model="kimi-k2.5:cloud",
            timeout=360,
        )
        agent = Agent(
            chat_generator=chat_generator,
            system_prompt="""You're a helpful AI agent expert on Skills designed by Anthropic. 
            When asked to generate skills you'll first read the example skills available using 
            the `read_example_skills` tool.
            You'll infer the intent of the user's question
            and will write the SKILL.md file with the `write_skill` tool. 
            Follow the example format from the tool.
            Remember that the SKILL.md file is for AI agents to read, not for humans.

            # Tools available

            ## Tool to read
            ### read_example_skills
            - Returns the content of the SKILL.md file used as an example. 
            - The SKILL.md file uses frontmatter Markdown style.

            ## Tool to write
            ### write_skill
            - Writes the resulting SKILL.md file the user asked using frontmatter Markdown style. 
            - Returns True if written succesfully, otherwise returns False.
            #### Parameters:
            - `dir_name`: An appropriate lowercase and one-word directory name.
            - `file_content`: The SKILL.md file content.
            """,
            tools=[read_example_skills, write_skill],
            exit_conditions=["text"]
        )
        return agent
