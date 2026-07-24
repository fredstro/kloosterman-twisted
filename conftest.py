"""
Test bootstrap.

Adds ``src/`` to ``sys.path`` so ``pytest`` works without installing the
package, and teaches stdlib ``doctest`` to understand Sage-style prompts so
that the ``EXAMPLES::`` blocks in the docstrings run under a plain Python
interpreter (i.e. without needing Sage).

Prompt rewrites applied on the fly:

    sage:    -> >>>
    ....:    -> ...

Skip markers (Sage convention):

    # optional - sage      -> skipped unless ``sage.all`` is importable
    # optional - cypari2   -> skipped unless ``cypari2`` is importable
"""

import doctest
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


# ---------------------------------------------------------------------------
# Sage-style prompt translation for stdlib doctest
# ---------------------------------------------------------------------------

_SAGE_PROMPT = re.compile(r"^(\s*)sage:\s?", re.MULTILINE)
_SAGE_CONT   = re.compile(r"^(\s*)\.\.\.\.:\s?", re.MULTILINE)


def _importable(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


_OPTIONAL_TAGS = {
    "sage":    lambda: _importable("sage.all"),
    "cypari2": lambda: _importable("cypari2"),
}
_OPTIONAL_RE = re.compile(r"#\s*optional\s*-\s*([\w.-]+)")


def _rewrite_optional(line: str) -> str:
    """Turn `# optional - X` into `# doctest: +SKIP` when X isn't importable."""
    m = _OPTIONAL_RE.search(line)
    if not m:
        return line
    tag = m.group(1)
    check = _OPTIONAL_TAGS.get(tag)
    if check is None or check():
        return line
    return _OPTIONAL_RE.sub("# doctest: +SKIP", line)


def _sage_to_python(text: str) -> str:
    text = _SAGE_PROMPT.sub(r"\1>>> ", text)
    text = _SAGE_CONT.sub(r"\1... ", text)
    text = "\n".join(_rewrite_optional(ln) for ln in text.split("\n"))
    return text


# Patch the parser at method level so it affects the *shared* DocTestParser
# instance that DocTestFinder binds at class-definition time.
_orig_parse = doctest.DocTestParser.parse
_orig_get_examples = doctest.DocTestParser.get_examples
_orig_get_doctest = doctest.DocTestParser.get_doctest


def _patched_parse(self, string, name="<string>"):
    return _orig_parse(self, _sage_to_python(string), name)


def _patched_get_examples(self, string, name="<string>"):
    return _orig_get_examples(self, _sage_to_python(string), name)


def _patched_get_doctest(self, string, globs, name, filename, lineno):
    return _orig_get_doctest(self, _sage_to_python(string), globs, name, filename, lineno)


doctest.DocTestParser.parse = _patched_parse
doctest.DocTestParser.get_examples = _patched_get_examples
doctest.DocTestParser.get_doctest = _patched_get_doctest
