"""Apply the unified diffs of this folder without git or patch.exe (Windows, plain Python).

    python tools\\apply_patch.py <Strata folder> patches\\01-....diff [more.diff ...]

The diffs were cut with normalized line endings, so the matcher compares lines with their endings
stripped and writes the file back with the endings it had (a new file gets CRLF: these are Windows
sources either way).  A hunk may sit up to 400 lines away from the line number it claims, which covers
the releases this was cut for; a file that moved further is a newer Strata, and then the hunk has to be
fixed against the new source rather than the applier.
"""
import sys
import pathlib


def parse_diff(path):
    """{new_path: (is_new, [(old_start, [(tag, line), ...]), ...])}; tag is ' ', '-' or '+'."""
    files = {}
    cur = None
    is_new = False
    lines = pathlib.Path(path).read_text(encoding="utf-8", errors="replace").splitlines(True)
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("--- "):
            old = line[4:].strip().split("\t")[0]
            new = lines[i + 1][4:].strip().split("\t")[0] if i + 1 < len(lines) and lines[i + 1].startswith("+++ ") else old
            name = (new if new != "/dev/null" else old)
            if name.startswith(("a/", "b/")):
                name = name[2:]
            is_new = old == "/dev/null"
            cur = name
            i += 2
            if is_new:                       # a new file: its lines follow the +++ header, with no @@ hunk
                body = []
                while i < len(lines) and lines[i].startswith("+"):
                    body.append(("+", lines[i][1:].rstrip("\r\n")))
                    i += 1
                files[cur] = (True, [(1, body)] if body else [])
                continue
            files[cur] = (False, [])
            continue
        if line.startswith("@@"):
            head = line.split("@@")[1].strip()
            start = int(head.split(" ")[0][1:].split(",")[0])
            body = []
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "--- ", "diff --git ")):
                tag = lines[i][:1]
                if tag in (" ", "-", "+"):
                    body.append((tag, lines[i][1:].rstrip("\r\n")))
                elif lines[i].strip() == "":
                    body.append((" ", ""))          # a bare newline in a diff is an empty context line
                elif lines[i].startswith("\\"):
                    pass                            # "\ No newline at end of file"
                else:
                    break
                i += 1
            files[cur][1].append((start, body))
            continue
        i += 1
    return files


def apply_hunks(have, hunks):
    """have: list of lines without endings.  Returns the new list, or None and the failed hunk."""
    have = list(have)
    shift = 0
    for start, body in hunks:
        want = [ln for tag, ln in body if tag in (" ", "-")]
        at = max(0, start - 1 + shift)
        lo, hi = max(0, at - 3000), min(len(have), at + 3000)
        found = -1
        for probe in range(lo, hi + 1):
            if have[probe:probe + len(want)] == want:
                found = probe
                break
        if found < 0:
            return None, (start, body)
        out = [ln for tag, ln in body if tag in (" ", "+")]
        have[found:found + len(want)] = out
        shift = found - (start - 1) + len(out) - len(want)
    return have, None


def apply_file(path, is_new, hunks):
    p = pathlib.Path(path)
    if is_new:
        have, crlf = [], True
    else:
        if not p.is_file():
            return f"{path}: no such file"
        raw = p.read_bytes()
        crlf = b"\r\n" in raw
        have = raw.decode("utf-8", "surrogateescape").replace("\r\n", "\n").split("\n")
        if have and have[-1] == "":
            have.pop()
    out, bad = apply_hunks(have, hunks)
    if out is None:
        start, body = bad
        where = body[0][1].strip()[:60] if body else ""
        return f"{path}: hunk at {start} does not match (context starts: {where!r})"
    eol = "\r\n" if crlf else "\n"
    text = eol.join(out) + eol
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(text.encode("utf-8", "surrogateescape"))
    return None


def main(argv):
    if len(argv) < 3:
        print("usage: apply_patch.py <Strata folder> <diff> [diff ...]", file=sys.stderr)
        return 2
    tree = pathlib.Path(argv[1])
    if not (tree / "src" / "qwen35").is_dir():
        print(f"{tree}: not a Strata source tree (no src/qwen35)", file=sys.stderr)
        return 2
    failed = 0
    for diff in argv[2:]:
        files = parse_diff(diff)
        print(f"== {pathlib.Path(diff).name}: {len(files)} file(s)")
        for name, (is_new, hunks) in files.items():
            err = apply_file(tree / name, is_new, hunks)
            if err:
                print("   " + err, file=sys.stderr)
                failed += 1
            else:
                print(f"   {name}: {len(hunks)} hunk(s){' (new file)' if is_new else ''}")
    if failed:
        print(f"\n{failed} file(s) failed; the tree is half-patched. Restore it from the release zip and retry.",
              file=sys.stderr)
        return 1
    print("\npatched. Build with:")
    print(f'  cmake -S "{tree}" -B "{tree}\\build-qwen35" -G Ninja -DCMAKE_BUILD_TYPE=Release '
          '-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=OFF '
          '-DSTRATA_Q6K_EXPERTS=ON -DSTRATA_GGML_DIR=<pinned llama.cpp>')
    print(f'  ninja -C "{tree}\\build-qwen35" strata strata_kq')
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
