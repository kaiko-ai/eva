"""EVA's main cli manager."""
import jsonargparse

from eva import interface 

def cli() -> object:
    """Main CLI factory."""
    return jsonargparse.CLI(interface.Interface)
