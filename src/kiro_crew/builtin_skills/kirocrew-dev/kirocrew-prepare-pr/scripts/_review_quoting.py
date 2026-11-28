"""Quote one argv word for the shell a printed spawn line is pasted into.

Stdlib only, and loaded by path (the scripts directory is not a package).
``local_review.py`` prints the reviewer spawn commands for PowerShell and for
cmd.exe; these two helpers quote each word so the pasted line runs as printed.
"""

#: Every character PowerShell reads as a single quote: the ASCII apostrophe and
#: U+2018 to U+201B. A doubled one stands for itself.
POWERSHELL_SINGLE_QUOTES = frozenset("'\u2018\u2019\u201a\u201b")


def powershell_quoted(word: str) -> str:
    return "'{}'".format("".join(ch * 2 if ch in POWERSHELL_SINGLE_QUOTES else ch for ch in word))


def cmd_quoted(word: str) -> str:
    # A backslash before the closing quote would escape it, so trailing ones are doubled.
    trailing = len(word) - len(word.rstrip("\\"))
    return '"{}{}"'.format(word, "\\" * trailing)
