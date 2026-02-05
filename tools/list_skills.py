import os
from haystack.tools import tool
from constants import SKILLS_DIR


@tool
def list_skills() -> list[str]:
    """
    Returns a list of existing SKILLs
    """
    try:
        skills = []
        for root, _, files in os.walk(SKILLS_DIR):
            for file in files:
                skills.append(os.path.join(root, file))

        return skills
    except Exception as e:
        return []
