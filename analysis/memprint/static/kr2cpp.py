"""Old C that does not parse as C++, rewritten so that it does, line for line (for the whole-program interpreter,
which parses every source as one C++ translation unit; the program is still built from the original).

    K&R definitions        float f(a, b) float a; int b; {  ->  float f(float a, int b) {
    implicit int           f(x) { ... }, extern n;           ->  int f(int x) { ... }, extern int n;
    old library prototypes extern char *strcpy();            ->  (removed: the C++ headers declare them)
    C++ keywords as names  float *new;                       ->  float *new_;
    register               register int i;                   ->  int i;

Only code outside strings and comments is changed, and every line keeps its number (declarations moved into a
function's header leave empty lines), so source locations in reports stay those of the original.
"""

import re

KEYWORDS = ("new", "delete", "this", "class", "template", "typename", "operator", "namespace", "private",
            "public", "protected", "virtual", "friend", "throw", "try", "catch", "mutable", "explicit")
STATEMENTS = {"if", "for", "while", "switch", "return", "sizeof", "do", "else", "case"}


def _code_mask(text):
    """Per character: whether it is code (not inside a string, character literal or comment)."""
    mask = [True] * len(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            j = min(j + 1, n)
        else:
            i += 1
            continue
        for k in range(i, j):
            mask[k] = False
        i = j
    return mask


def _depth(text, mask):
    """Brace depth before each character."""
    out, d = [], 0
    for ch, code in zip(text, mask):
        out.append(d)
        if code and ch == "{":
            d += 1
        elif code and ch == "}":
            d = max(0, d - 1)
    return out


def _params(decls, ids):
    """The parameter list for identifiers ids from K&R declarations ('float **w; int m, n;')."""
    types = {}
    for decl in decls.split(";"):
        decl = " ".join(decl.split())
        if not decl:
            continue
        m = re.match(r"^((?:(?:unsigned|signed|const|volatile|struct|union|enum|register|long|short)\s+)*\w+)\s*(.*)$",
                     decl)
        if not m:
            continue
        base, rest = m.group(1).replace("register ", ""), m.group(2)
        if not rest:                                                   # 'int x' read as base 'int', rest 'x'?
            continue
        for d in rest.split(","):
            d = d.strip()
            name = re.search(r"[A-Za-z_]\w*", d)
            if name:
                types[name.group(0)] = f"{base} {d}"
    return ", ".join(types.get(i, f"int {i}") for i in ids)


def convert(text, prototypes=None):
    """The text rewritten as C++ (see the module docstring); the same text if nothing applies. prototypes, a
    list, receives a declaration of every function rewritten (C calls a function before its definition)."""
    mask = _code_mask(text)

    def code_sub(pattern, repl, s, m_):
        out, last = [], 0
        for m in re.finditer(pattern, s):
            if all(m_[k] for k in range(m.start(), m.end())):
                out.append(s[last:m.start()])
                out.append(m.expand(repl) if isinstance(repl, str) else repl(m))
                last = m.end()
        out.append(s[last:])
        return "".join(out)

    for kw in KEYWORDS:                                                # C++ keywords used as names
        text = code_sub(rf"\b{kw}\b", f"{kw}_", text, mask)
        mask = _code_mask(text)
    text = code_sub(r"\bregister[ \t]+", "", text, mask)              # not a storage class in C++17
    mask = _code_mask(text)
    text = code_sub(r"(?m)^[ \t]*extern[ \t]+[\w \t\*]*\w[ \t]*\([ \t]*\)[ \t]*;", "", text, mask)
    mask = _code_mask(text)
    text = code_sub(r"(?m)^([ \t]*extern[ \t]+)([A-Za-z_]\w*)([ \t]*;)", r"\1int \2\3", text, mask)
    mask = _code_mask(text)
    depth = _depth(text, mask)
    header = re.compile(r"(?m)^(?P<ret>[A-Za-z_][\w \t\*]*?[ \t\*]+|)(?P<name>[A-Za-z_]\w*)[ \t]*"
                        r"\((?P<ids>[\w \t,\n]*)\)(?P<decls>(?:[^;{}()]*;)*?)(?P<sp>\s*)\{")
    out, last = [], 0
    for m in header.finditer(text):
        if depth[m.start()] != 0 or m.group("name") in STATEMENTS or not all(mask[k] for k in range(m.start(), m.end())):
            continue
        ret, ids, decls = m.group("ret"), [x.strip() for x in m.group("ids").split(",") if x.strip()], m.group("decls")
        if ret.strip() in ("", "static", "extern", "inline"):
            ret = (ret + "int ") if ret.strip() else "int "
        elif not ids and not decls.strip():
            continue                                                   # an ordinary f() definition
        if ids and not all(re.fullmatch(r"[A-Za-z_]\w*", i) for i in ids):
            continue                                                   # already a prototype (has types)
        if len(ids) == 1 and ids[0] == "void":
            continue
        params = _params(decls, ids)
        if prototypes is not None and "static" not in ret:
            prototypes.append(f"{ret.strip()} {m.group('name')}({params});")
        newlines = decls.count("\n") + m.group("ids").count("\n")
        out.append(text[last:m.start()])
        out.append(f"{ret}{m.group('name')}({params})" + "\n" * newlines + m.group("sp") + "{")
        last = m.end()
    out.append(text[last:])
    return "".join(out)


def convert_files(paths, out_dir):
    """Converted copies of the C files among paths that need it, in out_dir: {original path: copy}. Each copy also
    declares every function rewritten in any of them, on the first empty line after its includes."""
    import os
    texts, protos = {}, []
    for p in paths:
        if p.endswith(".c"):
            t = open(p, errors="replace").read()
            c = convert(t, protos)
            if c != t:
                texts[p] = c
    if texts:                                                          # functions the C calls that other
        called = set(re.findall(r"\b([A-Za-z_]\w*)\s*\(", " ".join(texts.values())))   # sources define
        for p in paths:
            if p.endswith(".c"):
                continue
            t = open(p, errors="replace").read()
            m_ = _code_mask(t)
            dep = _depth(t, m_)
            for m in re.finditer(r'(?m)^(?:extern\s+"C"\s+)?(?P<sig>(?P<ret>[A-Za-z_][\w \t\*]*?[ \t\*]+)'
                                 r'(?P<name>[A-Za-z_]\w*)\s*\((?P<params>[^;{)]*)\))\s*\{', t):
                if dep[m.start()] == 0 and m.group("name") in called and m.group("name") not in STATEMENTS and \
                        "__global__" not in m.group("ret") and "static" not in m.group("ret"):
                    protos.append(" ".join(m.group("sig").split()) + ";")
    out = {}
    import shutil
    shutil.rmtree(out_dir, ignore_errors=True)                         # ours: no copies left from an earlier run
    os.makedirs(out_dir, exist_ok=True)
    names = {re.search(r"([A-Za-z_]\w*)\(", pr).group(1) for pr in protos}
    for p in list(texts):                                              # the program's own headers they include:
        for inc in re.findall(r'(?m)^\s*#\s*include\s+"([^"]+)"', texts[p]):   # no f(); for an f declared fully
            h = os.path.join(os.path.dirname(p), inc)
            dst = os.path.join(out_dir, inc)
            if not os.path.exists(h) or os.path.exists(dst):
                continue
            t = open(h, errors="replace").read()
            m_ = _code_mask(t)

            def drop(m):
                return "" if m.group(1) in names else m.group(0)
            c = t
            for m in reversed(list(re.finditer(r"(?m)^[ \t]*(?:extern[ \t]+)?[\w \t\*]*?\b([A-Za-z_]\w*)[ \t]*\([ \t]*\)[ \t]*;",
                                               t))):
                if all(m_[k] for k in range(m.start(), m.end())):
                    c = c[:m.start()] + drop(m) + c[m.end():]
            c = convert(c)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            with open(dst, "w") as f:
                f.write(c)
    for p, c in texts.items():
        lines = c.split("\n")
        last_inc = max((i for i, l in enumerate(lines) if l.lstrip().startswith("#include")), default=-1)
        for i in range(last_inc + 1, len(lines)):
            if not lines[i].strip():
                lines[i] = " ".join(protos)
                break
        dst = os.path.join(out_dir, os.path.basename(p))
        with open(dst, "w") as f:
            f.write("\n".join(lines))
        out[p] = dst
    return out
