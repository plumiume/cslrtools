import importlib.metadata
from typing import Any, Callable
from clipar.entities import NamespaceWrapper

Info = tuple[
    NamespaceWrapper[Any],
    Callable[[Any], Any]
]
Plugins = dict[str, Info]

def load_plugins():

    entry_points = importlib.metadata.entry_points(
        group="cslrtools.dataset.v2.torch.plugins"
    )

    plugins: Plugins = {}

    for ep in entry_points:

        info = ep.load()
        plugins[ep.name] = info

    return plugins


