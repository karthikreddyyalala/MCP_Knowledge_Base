# Demo GIF for the README. Render with: brew install vhs && vhs docs/demo.tape
# Run from the repo root after `uv venv && uv pip install -e .` and one `kb index` (so models are cached).

Output docs/images/demo.gif
Set Shell "zsh"
Set FontSize 18
Set Width 1200
Set Height 640
Set Padding 24
Set TypingSpeed 35ms
Set Theme "Catppuccin Mocha"

Hide
Type "setopt interactive_comments && export PATH=$PWD/.venv/bin:$PATH && rm -rf /tmp/kb-demo && clear" Enter
Sleep 500ms
Show

Type "# index your docs (re-runs only touch changed files)" Enter
Type "kb index ./example-docs --db /tmp/kb-demo" Enter
Sleep 6s

Type "# a paraphrase: the doc says 'Tokens expire after 24 hours'" Enter
Type "kb search 'when does my login token stop working' --db /tmp/kb-demo --top-k 1" Enter
Sleep 4s

Type "# an exact technical string" Enter
Type "kb search 'multipart/form-data' --db /tmp/kb-demo --top-k 1" Enter
Sleep 4s

Type "# nothing relevant? it warns instead of pretending" Enter
Type "kb search 'best pizza in naples' --db /tmp/kb-demo --top-k 1" Enter
Sleep 4s

Type "# how well does search work on YOUR docs?" Enter
Type "kb eval ./example-docs --db /tmp/kb-demo --gold tests/data/example_gold_handwritten.jsonl" Enter
Sleep 12s
