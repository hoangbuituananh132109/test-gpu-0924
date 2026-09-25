"""Print a few basic details about the current Python environment."""

import platform
import sys


def summary() -> str:
    return f"Python {sys.version_info.major}.{sys.version_info.minor} on {platform.system()}"


if __name__ == "__main__":
    print(summary())
