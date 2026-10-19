"""std::string, std::map and std::ifstream for the C++ interpreter (interp_cpp), enough for programs that read their
options and input files with them (GraphBIG's argument parser and CSR loader).

Each object keeps a handle in its own first 8 bytes, so a byte-wise copy (a vector growing, a struct copied) copies
the object: a string's handle names an immutable text, a map's its Python dictionary, a stream's its file contents
and position. A string whose bytes are all zero is empty; one whose bytes are unknown has an unknown text.
"""

import os

import numpy as np

from . import clangast as ca
from .interp import UNK, Loc, Obj, Ptr, SENTINEL, _as_int

TAG = 1 << 24                                                          # handle = TAG * kind + index: 4 bytes, so
                                                                       # that copying an object unit by unit keeps it
STRING, MAP, FILE = 1, 2, 3
END = -2                                                               # a map iterator at the end
NPOS = 2 ** 64 - 1


def _name(t):
    s = t.get_canonical().spelling if not isinstance(t, str) else t
    s = s.replace("const ", "").replace("volatile ", "").strip()
    return s.replace("std::__cxx11::", "std::")


def is_string(t):
    return _name(t).startswith("std::basic_string<char")


def is_map(t):
    return _name(t).startswith("std::map<")


def is_map_iter(t):
    s = _name(t)
    return s.startswith("std::_Rb_tree_iterator<") or s.startswith("std::_Rb_tree_const_iterator<")


def is_ifstream(t):
    s = _name(t)
    return s.startswith("std::basic_ifstream<char") or s.startswith("std::basic_fstream<char")


def is_fpos(t):
    return _name(t).startswith("std::fpos<")


class TextLibrary:
    """Mixin for CppInterpreter."""

    # ------------------------------------------------------------ handles

    def _tables(self):
        return self.__dict__.setdefault("_text_tables", {STRING: [], MAP: [], FILE: []})

    def _handle_get(self, ptr, kind):
        if not isinstance(ptr, Ptr) or not isinstance(ptr.obj, Obj) or ptr.off is UNK or np.ndim(ptr.off):
            return None, False
        h = int(np.ravel(ptr.obj.load(int(ptr.off), 4))[0])
        if h == 0:
            return None, True                                          # all zero: a fresh (empty) object
        if h == SENTINEL or h // TAG != kind:
            return None, False
        return self._tables()[kind][h - kind * TAG], True

    def _handle_put(self, ptr, kind, value):
        if not isinstance(ptr, Ptr) or not isinstance(ptr.obj, Obj) or ptr.off is UNK or np.ndim(ptr.off):
            return
        t = self._tables()[kind]
        t.append(value)
        ptr.obj.store(int(ptr.off), kind * TAG + len(t) - 1, 4)

    # ------------------------------------------------------------ strings

    def str_get(self, v):
        """The text of a value: a std::string object, a string literal or a pointer to a known C string."""
        text = self._text(v)
        if text is not None:
            return text.split("\0")[0]
        if isinstance(v, Ptr):
            s, ok = self._handle_get(v, STRING)
            if ok:
                return s if s is not None else ""
        return None

    def str_put(self, ptr, text):
        if text is None:
            if isinstance(ptr, Ptr) and isinstance(ptr.obj, Obj) and ptr.off is not UNK and not np.ndim(ptr.off):
                ptr.obj.store(int(ptr.off), UNK, 4)
            return
        self._handle_put(ptr, STRING, text)

    def _cstring(self, text):
        """A pointer to a C string with this text (c_str)."""
        obj = Obj("c_str", "heap", len(text) + 1)
        obj.text = text
        for i, ch in enumerate(text.encode()):
            obj.store(i, ch, 1)
        obj.store(len(text), 0, 1)
        self.objects.append(obj)
        return Ptr(obj, 0)

    def _arg_text(self, a):
        return self.str_get(self.eval(a))

    def string_construct(self, dest, args_c):
        args_c = [a for a in args_c if "allocator<" not in a.type.spelling]
        if not args_c:
            self.str_put(dest, "")
        elif len(args_c) == 1:
            self.str_put(dest, self._arg_text(args_c[0]))
        else:
            first = self._arg_text(args_c[0])
            rest = [_as_int(self.eval(a)) for a in args_c[1:]]
            if first is None or any(x is UNK for x in rest):
                self.str_put(dest, None)
            elif len(rest) == 1 and not ca.is_pointer(args_c[0].type):  # (string, pos)
                self.str_put(dest, first[rest[0]:])
            elif len(rest) == 1:                                       # (const char *, n)
                self.str_put(dest, first[:rest[0]])
            else:                                                      # (string, pos, n)
                self.str_put(dest, first[rest[0]: rest[0] + rest[1]])
        return dest

    def string_method(self, name, obj, args_c, c, sret):
        me = self.str_get(obj)
        if name in ("size", "length"):
            return len(me) if me is not None else UNK
        if name == "empty":
            return int(me == "") if me is not None else UNK
        if name in ("c_str", "data"):
            return self._cstring(me) if me is not None else Ptr(UNK, UNK)
        if name in ("operator=", "assign"):
            self.str_put(obj, self._arg_text(args_c[0]) if args_c else None)
            return obj
        if name in ("operator+=", "append", "push_back"):
            a = self._arg_text(args_c[0]) if args_c else None
            if a is None and args_c:
                ch = _as_int(self.eval(args_c[0]))
                a = chr(ch) if ch is not UNK and not np.ndim(ch) else None
            self.str_put(obj, me + a if me is not None and a is not None else None)
            return obj
        if name == "clear":
            self.str_put(obj, "")
            return UNK
        if name == "substr":
            pos = _as_int(self.eval(args_c[0])) if args_c else 0
            n = _as_int(self.eval(args_c[1])) if len(args_c) > 1 else NPOS
            dest = sret if sret is not None else self._new_storage(c.type)
            ok = me is not None and pos is not UNK and n is not UNK and not np.ndim(pos) and not np.ndim(n)
            if ok and (n < 0 or n >= NPOS // 2):                       # npos (size_t -1, read signed): the rest
                n = len(me)
            self.str_put(dest, me[pos: pos + n] if ok else None)
            return dest
        if name in ("find", "rfind"):
            a = self._arg_text(args_c[0]) if args_c else None
            if me is None or a is None:
                return UNK
            i = me.find(a) if name == "find" else me.rfind(a)
            return i if i >= 0 else NPOS
        if name in ("operator[]", "at"):
            i = _as_int(self.eval(args_c[0])) if args_c else UNK
            return ord(me[i]) if me is not None and i is not UNK and not np.ndim(i) and 0 <= i < len(me) else \
                (0 if me is not None and i == len(me) else UNK)
        if name == "compare":
            a = self._arg_text(args_c[0]) if args_c else None
            return UNK if me is None or a is None else (me > a) - (me < a)
        return None

    def string_operator(self, name, args_c, c, sret):
        """A non-member operator on strings (==, !=, <, +), or None if the arguments are not strings."""
        if len(args_c) != 2:
            return None
        a, b = self._arg_text(args_c[0]), self._arg_text(args_c[1])
        if name == "operator+":
            dest = sret if sret is not None else self._new_storage(c.type)
            self.str_put(dest, a + b if a is not None and b is not None else None)
            return dest
        if a is None or b is None:
            return UNK
        return int({"operator==": a == b, "operator!=": a != b, "operator<": a < b, "operator>": a > b,
                    "operator<=": a <= b, "operator>=": a >= b}[name])

    # ------------------------------------------------------------ maps

    def _map(self, obj, value_type=None, create=True):
        m, ok = self._handle_get(obj, MAP)
        if m is None and ok and create:
            m = {"keys": [], "vals": {}, "vt": value_type}
            self._handle_put(obj, MAP, m)
        return m

    def _map_key(self, a):
        v = self.eval(a)
        s = self.str_get(v)
        if s is not None:
            return s
        v = _as_int(v)
        return None if v is UNK or np.ndim(v) else int(v)

    def map_method(self, name, obj, args_c, map_type, c, sret):
        t = map_type.get_canonical()
        vt = t.get_template_argument_type(1) if t.get_num_template_arguments() > 1 else None
        m = self._map(obj, vt)
        if m is None:
            return None
        if name == "operator[]":
            k = self._map_key(args_c[0])
            if k is None:
                return Ptr(UNK, UNK)
            if k not in m["vals"]:
                vt_ = m["vt"] or vt
                dest = self._new_storage(vt_)
                if dest is not None and vt_ is not None:
                    if is_string(vt_):
                        self.str_put(dest, "")
                    else:
                        self._default_construct(dest, vt_)
                m["keys"].append(k)
                m["vals"][k] = dest
            return m["vals"][k]
        if name in ("find", "end", "begin"):
            if name == "find":
                k = self._map_key(args_c[0])
                token = UNK if k is None else (m["keys"].index(k) if k in m["vals"] else END)
            else:
                token = END if name == "end" or not m["keys"] else 0
            dest = sret if sret is not None else self._new_storage(c.type)
            if isinstance(dest, Ptr) and isinstance(dest.obj, Obj):
                dest.obj.store(int(dest.off), token, 8)
            return dest
        if name == "count":
            k = self._map_key(args_c[0])
            return UNK if k is None else int(k in m["vals"])
        if name == "size":
            return len(m["keys"])
        if name == "empty":
            return int(not m["keys"])
        return None

    def map_iter_compare(self, name, args_c):
        a, b = (self.eval(x) for x in args_c[:2])
        if not (isinstance(a, Ptr) and isinstance(b, Ptr) and isinstance(a.obj, Obj) and isinstance(b.obj, Obj)):
            return UNK
        x, y = int(np.ravel(a.obj.load(int(a.off), 8))[0]), int(np.ravel(b.obj.load(int(b.off), 8))[0])
        if SENTINEL in (x, y):
            return UNK
        return int(x == y) if name == "operator==" else int(x != y)

    # ------------------------------------------------------------ input files

    def file_construct(self, dest, args_c):
        self._handle_put(dest, FILE, {"data": None, "pos": 0})
        if args_c:
            self.file_method("open", dest, args_c, None, None)
        return dest

    def file_method(self, name, obj, args_c, c, sret):
        f, ok = self._handle_get(obj, FILE)
        if f is None:
            if not ok:
                return None
            f = {"data": None, "pos": 0}
            self._handle_put(obj, FILE, f)
        if name == "open":
            path = self._arg_text(args_c[0]) if args_c else None
            f["data"], f["pos"] = None, 0
            if path is not None and os.path.isfile(path):
                with open(path, "rb") as fh:
                    f["data"] = fh.read()
            for a in args_c[1:]:
                self.eval(a)
            return UNK
        if name == "is_open":
            return int(f["data"] is not None)
        if name in ("good", "operator bool"):
            return int(f["data"] is not None and f["pos"] <= len(f["data"]))
        if name == "close":
            f["data"] = None
            return UNK
        if name == "seekg":
            off = _as_int(self.eval(args_c[0])) if args_c else UNK
            way = 0                                                    # beg 0, cur 1, end 2: by name (a static
            if len(args_c) > 1:                                        # member, fin.end, std::ios::end)
                words = [t.spelling for t in args_c[1].get_tokens()]
                way = {"beg": 0, "cur": 1, "end": 2}.get(words[-1] if words else "", None)
                if way is None:
                    way = _as_int(self.eval(args_c[1]))
            if f["data"] is None or off is UNK or way is UNK:
                return obj
            base = {0: 0, 1: f["pos"], 2: len(f["data"])}.get(way, 0)
            f["pos"] = max(0, base + off)
            return obj
        if name == "tellg":
            dest = sret if sret is not None else self._new_storage(c.type)
            pos = f["pos"] if f["data"] is not None else -1
            if isinstance(dest, Ptr) and isinstance(dest.obj, Obj):
                dest.obj.store(int(dest.off), pos, 8)
            return dest
        if name == "read":
            ptr, n = self.eval(args_c[0]), _as_int(self.eval(args_c[1]))
            if f["data"] is None or n is UNK or not isinstance(ptr, Ptr) or not isinstance(ptr.obj, Obj) or \
                    ptr.off is UNK or np.ndim(ptr.off):
                return obj
            chunk = f["data"][f["pos"]: f["pos"] + int(n)]
            f["pos"] += len(chunk)
            self._write_bytes(ptr, chunk)
            return obj
        return None

    def _write_bytes(self, ptr, data):
        """Bytes read from a file into memory: 8-byte values at 8-byte offsets (with their upper halves in the
        units after them, for 4-byte reads), what is left byte by byte."""
        base = int(ptr.off)
        n8 = len(data) // 8
        if n8:
            v = np.frombuffer(data[: 8 * n8], dtype="<u8").astype(np.int64)
            offs = base + 8 * np.arange(n8, dtype=np.int64)
            ptr.obj.store(offs + 4, (v >> 32) & 0xFFFFFFFF, 4)
            ptr.obj.store(offs, v, 8)
        for i in range(8 * n8, len(data)):
            ptr.obj.store(base + i, data[i], 1)
        ptr.obj.nbytes = max(ptr.obj.nbytes, base + len(data))

    def fpos_value(self, obj):
        if isinstance(obj, Ptr) and isinstance(obj.obj, Obj) and obj.off is not UNK and not np.ndim(obj.off):
            v = int(np.ravel(obj.obj.load(int(obj.off), 8))[0])
            return UNK if v == SENTINEL else v
        return UNK
