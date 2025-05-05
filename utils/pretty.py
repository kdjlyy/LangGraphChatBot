from dataclasses import dataclass
import rich
from rich.rule import Rule
from rich.text import Text
from rich import print as rprint

from rich.console import Console

RICH_CONSOLE = Console()


@dataclass
class ALogger:
    prefix: str = ""

    def title(self, text: str | Text, rule_style="bright_black"):
        lits = []
        if self.prefix:
            lits.append(rich.markup.escape(self.prefix))
        if text:
            lits.append(text)
        rprint(Rule(title=" ".join(lits), style=rule_style))


def log_title(text: str | Text, rule_style="bright_black"):
    objs = []
    if text:
        objs.append(Rule(title=text, style=rule_style))
    rprint(*objs)


if __name__ == "__main__":
    ALogger("[utils]").title("Hello World!")
