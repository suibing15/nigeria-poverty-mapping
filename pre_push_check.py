"""
pre_push_check.py - Run this BEFORE every `git push` to make sure no DHS data or derived cluster tables leave your computer.

USAGE (inside your project folder, .venv activated):
    python pre_push_check.py          checks staged files and UNPUSHED commits (use before every push)
    python pre_push_check.py --all    checks the ENTIRE history, including commits already on GitHub

IT CHECKS 3 THINGS
  1. STAGED files (what the next commit would contain).
  2. UNPUSHED commits: every file ever added in commits that are not yet on GitHub. This catches data that was
     committed earlier and deleted later, which would still be uploaded with the history.
  3. FILE SIZE: anything above 5 MB is flagged (GitHub rejects files above 100 MB, and data tables are usually large).

WHAT IS FLAGGED
  - DHS or GIS data formats: .dta .shp .shx .dbf .sav .zip .parquet .pkl .joblib
  - anything inside data/ or outputs/
  - cluster tables: file names containing 'features_dhs', 'dhs_cluster' or 'cluster_wealth'
  - secrets: names containing '.env', 'credentials', 'token' or 'secret'

It prints OK TO PUSH only when none of these is found. It never changes anything.
"""

import os
import re
import subprocess
import sys

BAD_EXT = {".dta", ".shp", ".shx", ".dbf", ".sav", ".zip", ".parquet", ".pkl", ".joblib"}
BAD_DIRS = ("data/", "outputs/")
TABLE_NAME = re.compile(r"(features_dhs|dhs_cluster|cluster_wealth)", re.I)   # cluster tables: only checked on non-code files
SECRET_NAME = re.compile(r"(\.env$|credential|token|secret)", re.I)
CODE_EXT = {".py", ".md", ".txt", ".ps1"}                                      # code and documents are never cluster tables
MAX_MB = 5


def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def problem(path):
    p = path.replace("\\", "/")
    reasons = []
    if os.path.splitext(p)[1].lower() in BAD_EXT:
        reasons.append("data or archive format")
    base, ext = os.path.basename(p), os.path.splitext(p)[1].lower()
    if p.startswith(BAD_DIRS) and ext not in CODE_EXT:      # code and notes inside data/ are fine; data files are not
        reasons.append("inside data/ or outputs/")
    if ext not in CODE_EXT and TABLE_NAME.search(base):
        reasons.append("cluster table by name")
    if SECRET_NAME.search(base):
        reasons.append("possible secret by name")
    return reasons


def main():
    rc, top, err = git("rev-parse", "--show-toplevel")
    if rc != 0:
        sys.exit("ERROR: this is not a git repository. Run it inside your project folder.")
    os.chdir(top)
    found = []

    # 1. staged files
    _, out, _ = git("diff", "--cached", "--name-only", "--diff-filter=ACMR")
    staged = [x for x in out.splitlines() if x]
    for f in staged:
        for r in problem(f):
            found.append(("STAGED", f, r))
        if os.path.isfile(f) and os.path.getsize(f) > MAX_MB * 1024 * 1024:
            found.append(("STAGED", f, f"larger than {MAX_MB} MB ({os.path.getsize(f) / 1e6:.1f} MB)"))

    # 2. unpushed history (or all history if there is no upstream yet)
    rc, _, _ = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if "--all" in sys.argv:
        scope, args = "ENTIRE history (including what is already on GitHub)", ["log", "--all", "--name-only", "--pretty=format:%h", "--diff-filter=A"]
    elif rc == 0:
        scope, args = "unpushed commits", ["log", "@{u}..HEAD", "--name-only", "--pretty=format:%h", "--diff-filter=A"]
    else:
        scope, args = "all commits (no upstream set)", ["log", "--name-only", "--pretty=format:%h", "--diff-filter=A"]
    _, hist, _ = git(*args)
    commit = ""
    seen = set()
    for line in hist.splitlines():
        line = line.strip()
        if not line:
            continue
        if re.fullmatch(r"[0-9a-f]{7,40}", line):
            commit = line
            continue
        for r in problem(line):
            if (line, r) not in seen:
                seen.add((line, r))
                found.append((f"HISTORY ({commit})", line, r))

    print(f"Checked: {len(staged)} staged file(s) and the {scope}.")
    if not found:
        print("\nOK TO PUSH. No data files, cluster tables, secrets or large files found.")
        return
    print("\nDO NOT PUSH YET. Found:")
    for where, f, r in found:
        print(f"  [{where}] {f}  ({r})")
    if any(w.startswith("HISTORY") for w, _, _ in found):
        print("\nSome of these are in your commit history. Deleting the file in a new commit does NOT remove it from "
              "history. It has to be purged (git filter-repo) before it is safe.")
    else:
        print("\nUnstage with:  git restore --staged <file>   and make sure the file is covered by .gitignore.")
    sys.exit(1)


if __name__ == "__main__":
    main()
