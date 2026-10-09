"""
Console output helpers, so every pipeline step prints in the same style.

Importing this module also sets up the output for the whole process: stdout and stderr
are switched to UTF-8 with line buffering, Python warnings are printed through warn(),
and uncaught exceptions get a one-line error before their traceback.

Conventions used across the pipeline:
  header()   one per step, at the top
  section()  sub-heading inside a step (sentence case, no trailing period)
  info()     neutral status; ongoing work ends with "..."
  success()  something finished or was loaded
  warn()     recoverable problem, full sentence ending with "."
  error()    fatal problem, full sentence ending with "." plus what to do next
  kv()       aligned "label  value" lines (summaries, metrics, saved paths)
  ask()      "? question ›" prompt; only used when can_ask() is True
  done()     one per step, at the bottom
"""
import os
import sys
import warnings

# Make sure the symbols below never crash the output (e.g. when redirected to a file on Windows),
# and flush every line so messages stay in order when the output is piped or logged.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

WIDTH = 64
INDENT = "  "
KV_WIDTH = 24

_progress_active = False
_progress_interactive = sys.stdout.isatty()


def _end_progress_line() -> None:
    """Moves past an in-place progress line so the next message starts on a fresh line."""
    global _progress_active
    if _progress_active:
        print()
        _progress_active = False


def _rule(char: str) -> None:
    """Prints a full-width horizontal line."""
    print(char * WIDTH)


def banner(title: str) -> None:
    """Top-level banner, used only by run_all.py."""
    _end_progress_line()
    print()
    _rule("═")
    print(f" {title}")
    _rule("═")


def header(title: str) -> None:
    """Step banner. Shows 'STEP x/y' when launched from run_all.py."""
    _end_progress_line()
    step = os.environ.get("XPASS_STEP")
    label = f"STEP {step} · {title}" if step else title
    print()
    _rule("━")
    print(f" {label}")
    _rule("━")


def section(title: str) -> None:
    """Smaller sub-heading inside a step."""
    _end_progress_line()
    print(f"\n{INDENT}▸ {title}")


def info(msg: str) -> None:
    """Neutral status line:  › msg"""
    _end_progress_line()
    print(f"{INDENT}› {msg}")


def success(msg: str) -> None:
    """Something finished, loaded or was saved:  ✔ msg"""
    _end_progress_line()
    print(f"{INDENT}✔ {msg}")


def warn(msg: str) -> None:
    """A problem the step can recover from:  ⚠ msg"""
    _end_progress_line()
    print(f"{INDENT}⚠ {msg}")


def error(msg: str) -> None:
    """A problem that stops the step:  ✖ msg"""
    _end_progress_line()
    print(f"{INDENT}✖ {msg}")


def kv(label: str, value, width: int = KV_WIDTH) -> None:
    """Aligned 'label  value' line, used for summaries."""
    _end_progress_line()
    print(f"{INDENT}  {label:<{width}} {value}")


def listing(items, label: str = None) -> None:
    """Prints a long list (e.g. features) wrapped to the console width."""
    _end_progress_line()
    if label:
        print(f"{INDENT}› {label} ({len(items)}):")
    # Wrap on item boundaries only, so names containing spaces are never split
    line = ""
    for item in (str(i) for i in items):
        candidate = f"{line}, {item}" if line else item
        if line and len(candidate) > WIDTH - 6:
            print(f"{INDENT}    {line},")
            line = item
        else:
            line = candidate
    if line:
        print(f"{INDENT}    {line}")


def can_ask() -> bool:
    """True only when a person is at the terminal, so automated runs never block on a prompt."""
    return sys.stdin is not None and sys.stdin.isatty()


def ask(question: str) -> str:
    """Prompts the user and returns the stripped answer."""
    _end_progress_line()
    return input(f"{INDENT}? {question} › ").strip()


def ask_yes_no(question: str) -> bool:
    """Prompts a [Y/N] question until it gets a valid answer."""
    while True:
        answer = ask(f"{question} [Y/N]").upper()
        if answer in ("Y", "N"):
            return answer == "Y"
        warn("Please answer Y or N.")


def progress(done: int, total: int, label: str = "") -> None:
    """
    Progress line. In a terminal it updates in place; when the output is piped or logged
    it prints a plain line every 10% instead of thousands of carriage returns.
    """
    global _progress_active
    pct = done / total * 100 if total else 100
    digits = len(f"{total:,}")
    line = f"{INDENT}[ {done:>{digits},}/{total:,} ] {pct:5.1f}%  {label}"[:WIDTH]

    if _progress_interactive:
        sys.stdout.write("\r" + line.ljust(WIDTH))
        sys.stdout.flush()
        _progress_active = True
    else:
        step = max(1, total // 10)
        if done == total or done % step == 0:
            print(line)


def done(msg: str) -> None:
    """Closing line of a step."""
    _end_progress_line()
    print(f"\n{INDENT}✔ {msg}")
    _rule("─")


def describe_scope(league: str = None, season: str = None) -> str:
    """Human-readable --league / --season scope, shared by every step that uses it."""
    if not league:
        return "all leagues · all seasons"
    return f"{league} · {season or 'all seasons'}"


# Route Python warnings (ours and the libraries') through warn(), so they appear in place
# and in the same style instead of as raw "file.py:123: UserWarning: ..." lines on stderr.
def _show_warning(message, category, filename, lineno, file=None, line=None):
    prefix = "" if category is UserWarning else f"{category.__name__}: "
    warn(f"{prefix}{message}")


warnings.showwarning = _show_warning


# Unexpected crashes: print a styled one-line error first, then the full traceback for debugging.
def _excepthook(exc_type, exc, tb):
    if issubclass(exc_type, KeyboardInterrupt):
        error("Interrupted.")
        return
    error(f"Unexpected {exc_type.__name__}: {exc}")
    sys.__excepthook__(exc_type, exc, tb)


sys.excepthook = _excepthook
