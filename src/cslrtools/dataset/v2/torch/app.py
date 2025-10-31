
from clipar import namespace, mixin

from .plugins import load_plugins

def main() -> int:

    plugins = load_plugins()

    @namespace
    class AppArgs(mixin.ReprMixin, mixin.CommandMixin):
        """Application arguments for dataset handling with PyTorch."""

    for name, (ns, _) in plugins.items():
        AppArgs.add_wrapper(name, ns)

    args = AppArgs.parse_args()

    if args.command is None:
        print(
            "No command specified. Use --help to see available commands."
        )
        return 1

    plugins[args.command][1](args)

    return 0

if __name__ == "__main__":
    exit(main())
