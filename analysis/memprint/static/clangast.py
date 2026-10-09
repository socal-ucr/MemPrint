"""libclang helpers: parsing, operator kinds, constant evaluation and for-loop parts.

The Python bindings of libclang 18 do not expose operator kinds or constant
evaluation, so they are called through the C API.
"""

import ctypes
import subprocess
from functools import lru_cache

import clang.cindex as ci

K = ci.CursorKind

BINARY_OPS = {
    3: "*", 4: "/", 5: "%", 6: "+", 7: "-", 8: "<<", 9: ">>", 11: "<", 12: ">", 13: "<=", 14: ">=",
    15: "==", 16: "!=", 17: "&", 18: "^", 19: "|", 20: "&&", 21: "||", 22: "=", 23: "*=", 24: "/=",
    25: "%=", 26: "+=", 27: "-=", 28: "<<=", 29: ">>=", 30: "&=", 31: "^=", 32: "|=", 33: ",",
}
UNARY_OPS = {1: "post++", 2: "post--", 3: "pre++", 4: "pre--", 5: "&", 6: "*", 7: "+", 8: "-", 9: "~", 10: "!"}

ARRAY_TYPES = (ci.TypeKind.CONSTANTARRAY, ci.TypeKind.INCOMPLETEARRAY, ci.TypeKind.VARIABLEARRAY,
               ci.TypeKind.DEPENDENTSIZEDARRAY)
FLOAT_TYPES = (ci.TypeKind.FLOAT, ci.TypeKind.DOUBLE, ci.TypeKind.LONGDOUBLE, ci.TypeKind.FLOAT128,
               ci.TypeKind.HALF)

_lib = None

# The interpreters ask libclang the same questions about the same nodes many times (every iteration of a
# loop that runs one point at a time). Each answer is a ctypes call, and get_children builds new cursor
# objects through a callback, so caching answers per cursor object only pays if the same objects come
# back: child lists are therefore cached (on the cursor, and by hash for cursors created elsewhere, checked
# for equality), and kinds and canonical types are cached on the objects that computed them.
_CHILDREN = {}
_get_children = ci.Cursor.get_children
_kind = ci.Cursor.kind.fget
_canonical = ci.Type.get_canonical


def _cached_children(self):
    lst = self.__dict__.get("_children")
    if lst is None:
        key = (ci.conf.lib.clang_hashCursor(self), self._kind_id)
        hit = _CHILDREN.get(key)
        if hit is not None and hit[0] == self:
            lst = hit[1]
        else:
            lst = list(_get_children(self))
            _CHILDREN[key] = (self, lst)
        self._children = lst
    return iter(lst)


def _cached_kind(self):
    k = self.__dict__.get("_kind_cached")
    if k is None:
        k = self._kind_cached = _kind(self)
    return k


def _cached_canonical(self):
    c = self.__dict__.get("_canonical")
    if c is None:
        c = self._canonical = _canonical(self)
    return c


ci.Cursor.get_children = _cached_children
ci.Cursor.kind = property(_cached_kind)
ci.Type.get_canonical = _cached_canonical


def lib():
    global _lib
    if _lib is None:
        ci.Index.create()  # loads the library
        _lib = ci.conf.lib
        _lib.clang_getCursorBinaryOperatorKind.argtypes = [ci.Cursor]
        _lib.clang_getCursorBinaryOperatorKind.restype = ctypes.c_int
        _lib.clang_getCursorUnaryOperatorKind.argtypes = [ci.Cursor]
        _lib.clang_getCursorUnaryOperatorKind.restype = ctypes.c_int
        _lib.clang_Cursor_Evaluate.argtypes = [ci.Cursor]
        _lib.clang_Cursor_Evaluate.restype = ctypes.c_void_p
        _lib.clang_EvalResult_getKind.argtypes = [ctypes.c_void_p]
        _lib.clang_EvalResult_getKind.restype = ctypes.c_int
        _lib.clang_EvalResult_getAsLongLong.argtypes = [ctypes.c_void_p]
        _lib.clang_EvalResult_getAsLongLong.restype = ctypes.c_longlong
        _lib.clang_EvalResult_getAsDouble.argtypes = [ctypes.c_void_p]
        _lib.clang_EvalResult_getAsDouble.restype = ctypes.c_double
        _lib.clang_EvalResult_dispose.argtypes = [ctypes.c_void_p]
    return _lib


def binary_op(cursor):
    return BINARY_OPS.get(lib().clang_getCursorBinaryOperatorKind(cursor))


def compound_op(cursor):
    """The operator of a compound assignment ('+=' -> '+')."""
    return binary_op(cursor)[:-1]


def unary_op(cursor):
    return UNARY_OPS.get(lib().clang_getCursorUnaryOperatorKind(cursor))


def evaluate(cursor):
    """The value of a constant expression (int or float), or None."""
    result = lib().clang_Cursor_Evaluate(cursor)
    if not result:
        return None
    try:
        kind = lib().clang_EvalResult_getKind(result)
        if kind == 1:
            return int(lib().clang_EvalResult_getAsLongLong(result))
        if kind == 2:
            return float(lib().clang_EvalResult_getAsDouble(result))
        return None
    finally:
        lib().clang_EvalResult_dispose(result)


@lru_cache(maxsize=None)
def _gcc_include():
    try:
        out = subprocess.run(["cc", "-print-file-name=include"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def parse(path, defines=(), includes=(), extra=(), cplusplus=False):
    """Parse a source file. Errors are kept in tu.diagnostics; the AST is used as far as it goes."""
    args = [f"-D{d}" for d in defines] + [f"-I{i}" for i in includes] + list(extra)
    if _gcc_include():
        args.append(f"-I{_gcc_include()}")
    if cplusplus:
        args = ["-x", "c++", "-std=c++17"] + args
    return ci.Index.create().parse(str(path), args=args)


def children(cursor):
    return list(cursor.get_children())


def strip(cursor):
    """Skip implicit casts and parentheses."""
    while cursor.kind in (K.UNEXPOSED_EXPR, K.PAREN_EXPR) and len(children(cursor)) == 1:
        cursor = children(cursor)[0]
    return cursor


def is_array(type_):
    return type_.get_canonical().kind in ARRAY_TYPES


def is_float(type_):
    return type_.get_canonical().kind in FLOAT_TYPES


def is_pointer(type_):
    return type_.get_canonical().kind == ci.TypeKind.POINTER


def type_size(type_):
    """sizeof(type) in bytes, or None (incomplete or variable-length)."""
    size = type_.get_canonical().get_size()
    return size if size is not None and size > 0 else None


def pointee_size(type_):
    canonical = type_.get_canonical()
    if canonical.kind == ci.TypeKind.POINTER:
        return type_size(canonical.get_pointee())
    if canonical.kind in ARRAY_TYPES:
        return type_size(canonical.element_type)
    return None


def for_parts(cursor):
    """(init, cond, inc, body) of a FOR_STMT; missing parts are None.

    libclang lists only the parts that exist, so they are told apart by their
    position relative to the two semicolons in the loop header."""
    kids = children(cursor)
    body = kids[-1]
    header = kids[:-1]
    if len(header) == 3:
        return header[0], header[1], header[2], body
    semis = []
    depth = 0
    for token in cursor.get_tokens():
        if token.spelling == "(":
            depth += 1
        elif token.spelling == ")":
            depth -= 1
            if depth == 0:
                break
        elif token.spelling == ";" and depth == 1:
            semis.append(token.extent.start.offset)
    parts = [None, None, None]
    for kid in header:
        start = kid.extent.start.offset
        slot = 0 if len(semis) < 1 or start < semis[0] else 1 if len(semis) < 2 or start < semis[1] else 2
        parts[slot] = kid
    return parts[0], parts[1], parts[2], body
