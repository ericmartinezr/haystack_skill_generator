import os
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack.tools import tool
from haystack_integrations.components.generators.ollama import OllamaChatGenerator


EXAMPLE_SKILLS_DIR = "example_skills"
SKILLS_DIR = "skills"


@tool
def read_example_skills() -> str:
    """
    Reads the example skills for adding to LLM's context

    Returns:
    - Returns the content of the file
    """
    for root, _, files in os.walk(EXAMPLE_SKILLS_DIR):
        for file in files:
            skill_path = os.path.join(root, file)
            with open(skill_path, "r") as f:
                return f.read()


@tool
def write_skill(dir_name: str, file_content: str) -> bool:
    """
    Writes a SKILL.md file using frontmatter Markdown style

    Arguments:
    - dir_name (str): The directory where the SKILL.md will be placed
    - file_content (str): The content of the SKILL.md file

    Returns:
    - If written succesfully returns True, otherwise returns FALSE
    """
    try:
        # Creates the directoy
        dir_path = os.path.join(SKILLS_DIR, dir_name)
        os.makedirs(dir_path, exist_ok=False)

        # Writes the file
        file_path = os.path.join(dir_path, "SKILL.md")
        with open(file_path, "w") as f:
            content_written = f.write(file_content)

        return content_written > 0
    except Exception as e:
        return False


chat_generator = OllamaChatGenerator(model="kimi-k2.5:cloud")
agent = Agent(
    chat_generator=chat_generator,
    system_prompt="""You're a helpful AI agent. 
    When asked to generate skills you'll first read the example skills available using 
    the `read_example_skills` tool.
    After reading the skills you'll infer the intent of the user's question
    and will write the SKILL.md file the user requires using the example's format using the `write_skill` tool.

    # Tools available
    ## Tool to read
    `read_example_skills`: Returns the content of the SKILL.md file used as an example. The SKILL.md file uses frontmatter Markdown style.
    ## Tool to write
    `write_skill`: Writes the resulting SKILL.md file the user asked using frontmatter Markdown style. Returns True if written succesfully, otherwise returns False.
    ### Parameters:
    - `dir_name`: An appropriate directory name for the SKILL.md file.
    - `file_content`: The SKILL.md file content.
    """,
    tools=[read_example_skills, write_skill]
)

user_message = ChatMessage.from_user(
    "Write a SKILL.md file for AI's to read to assist on writing Machine Learning"
)
response = agent.run(messages=[user_message])
