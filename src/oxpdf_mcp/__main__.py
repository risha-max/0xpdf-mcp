"""CLI entry: `oxpdf-mcp` or `python -m oxpdf_mcp`."""

from oxpdf_mcp.server import create_mcp_server


def main() -> None:
    create_mcp_server().run()


if __name__ == "__main__":
    main()
