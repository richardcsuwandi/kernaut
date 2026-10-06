"""Build the documentation example from the packaged archive viewer."""

from pathlib import Path

from mkdocs.structure.files import File


def on_files(files, config):
    static = Path(__file__).resolve().parents[2] / "src/kernaut/viz/static"
    page = (static / "index.html").read_text()
    page = page.replace(
        "</head>",
        '<meta name="kernaut-snapshot" content="snapshot.json">'
        '<link rel="stylesheet" href="guide.css"></head>',
    )
    page = page.replace(
        "<body>",
        '<body class="demo-mode"><section id="demo-guide" aria-label="Archive walkthrough">'
        "<p>Loading the interactive archive example…</p></section>"
        "<noscript>This archive walkthrough requires JavaScript. "
        "The documentation also describes each viewer section.</noscript>",
    )
    page = page.replace(
        '<script src="app.js"></script>',
        '<script src="guide.js"></script><script src="app.js"></script>',
    )
    page = page.replace(
        "<title>Kernaut, Archive</title>", "<title>Kernaut, Interactive archive example</title>"
    )
    files.append(File.generated(config, "demo/index.html", content=page))
    for name in ("style.css", "app.js"):
        files.append(File.generated(config, f"demo/{name}", content=(static / name).read_text()))
    return files
