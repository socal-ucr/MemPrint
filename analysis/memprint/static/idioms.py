"""Access idioms of a program's memory references, from its source alone.

For code the interpreter cannot run (input-dependent control, C++, MPI), the
references are classified by how their address is formed:

    affine     subscripts built from integer variables, constants, + - and * by a constant
    indirect   a subscript that itself loads memory or calls a function (A[B[i]], CSR neighbours)
    pointer    dereference of a pointer or ->member (pointer chasing, iterators)
    other      any other subscript (i * j, i % n, floating-point)

and the loops by what bounds them:

    counted    for loop whose bound needs no memory load
    data       for loop bounded by a load or call (e < row[v + 1], it != v.end())
    while      while / do loops

Each reference is weighted by 10^(loop depth), a static stand-in for its
dynamic count. The affine share is the confidence that a program behaves like
the affine kernels the static spectrum is exact for.
"""

from pathlib import Path

import numpy as np

from . import clangast as ca
from .clangast import K

LOOPS = (K.FOR_STMT, K.WHILE_STMT, K.DO_STMT, K.CXX_FOR_RANGE_STMT)
MEMORY = (K.ARRAY_SUBSCRIPT_EXPR, K.MEMBER_REF_EXPR, K.CALL_EXPR)
LOOKUPS = {"find", "count", "at", "insert", "emplace", "erase", "lower_bound", "upper_bound", "contains"}
ALLOCATING = {"malloc", "calloc", "realloc", "push_back", "emplace_back", "resize", "reserve", "insert",
              "operator new", "operator new[]", "make_shared", "make_unique"}
FEATURES = ["affine", "indirect", "pointer", "other", "loops_counted", "loops_data", "loops_while",
            "max_depth", "alloc_in_loop", "references"]


def _in_project(cursor, root):
    f = cursor.location.file
    return f is not None and str(f.name).startswith(str(root))


def _loads_memory(expr):
    """Does evaluating expr load memory (a subscript, a dereference) or call a function?"""
    stack = [expr]
    while stack:
        n = stack.pop()
        if n.kind == K.ARRAY_SUBSCRIPT_EXPR or n.kind == K.CALL_EXPR:
            return True
        if n.kind == K.UNARY_OPERATOR and ca.unary_op(n) == "*":
            return True
        if n.kind == K.MEMBER_REF_EXPR and ca.children(n) and ca.is_pointer(ca.children(n)[0].type) \
                and ca.strip(ca.children(n)[0]).kind != K.CXX_THIS_EXPR:
            return True
        stack.extend(n.get_children())
    return False


def _index_class(expr):
    if _loads_memory(expr):
        return "indirect"
    stack = [expr]
    while stack:
        n = stack.pop()
        if n.kind == K.BINARY_OPERATOR:
            op = ca.binary_op(n)
            if op == "*":
                a, b = ca.children(n)
                if ca.evaluate(a) is None and ca.evaluate(b) is None:
                    return "other"
            elif op not in ("+", "-", "<<"):
                return "other"
        elif n.kind == K.FLOATING_LITERAL or (n.kind.is_expression() and ca.is_float(n.type)):
            return "other"
        stack.extend(n.get_children())
    return "affine"


def _names(expr):
    return {d.spelling for d in _decl_refs(expr)} if expr is not None else set()


def _varies(expr, loop_vars):
    """Does expr load memory or call a function at an address that depends on an enclosing loop?"""
    return _loads_memory(expr) and bool(_names(expr) & loop_vars)


def _bound_from_data(cond, function, loop_vars, own):
    """Is the loop bounded by data: the condition, or a variable in it, is a load or call that
    depends on an enclosing loop variable (row[v + 1], or edge_range(v, e0, e1) filling e1)?
    Loads that do not depend on an enclosing loop are parameters (net->batch) and do not count."""
    if _varies(cond, loop_vars):
        return True
    names = _names(cond) - own
    if not names or function is None or not loop_vars:
        return False
    stack = [function]
    while stack:
        n = stack.pop()
        if n.kind == K.VAR_DECL and n.spelling in names:
            kids = ca.children(n)
            if kids and any(t.spelling == "=" for t in n.get_tokens()) and _varies(kids[-1], loop_vars):
                return True
        if n.kind == K.BINARY_OPERATOR and ca.binary_op(n) == "=":
            lhs, rhs = ca.children(n)
            if ca.strip(lhs).kind == K.DECL_REF_EXPR and ca.strip(lhs).spelling in names and _varies(rhs, loop_vars):
                return True
        if n.kind == K.CALL_EXPR:
            args = list(n.get_arguments())
            callee = n.referenced
            params = list(callee.get_arguments()) if callee is not None else []
            out_args = set()
            for i, arg in enumerate(args):
                a = ca.strip(arg)
                if a.kind == K.UNARY_OPERATOR and ca.unary_op(a) == "&":
                    a = ca.strip(ca.children(a)[0])
                elif not (i < len(params) and _out_reference(params[i].type)):
                    continue
                if a.kind == K.DECL_REF_EXPR:
                    out_args.add(a.spelling)
            if out_args & names and set().union(*(_names(a) for a in args)) & loop_vars:
                return True
        stack.extend(n.get_children())
    return False


def _out_reference(type_):
    """A non-const C++ lvalue reference parameter (the callee can write the argument)."""
    return type_.kind == ca.ci.TypeKind.LVALUEREFERENCE and not type_.get_pointee().is_const_qualified()


def _decl_refs(expr):
    out, stack = [], [expr]
    while stack:
        n = stack.pop()
        if n.kind == K.DECL_REF_EXPR and n.referenced is not None and n.referenced.kind in (K.VAR_DECL, K.PARM_DECL):
            out.append(n)
        stack.extend(n.get_children())
    return out


def _is_map(cursor):
    kids = ca.children(cursor)
    spelling = (kids[0].type.spelling if kids else "") + " " + cursor.type.spelling
    return "map" in spelling or "set" in spelling or "hash" in spelling


def _worst(classes):
    for c in ("indirect", "other", "affine"):
        if c in classes:
            return c
    return "affine"


def _subscript_chain(n):
    """Index expressions of a[i][j]... from the outermost subscript."""
    idx = []
    while n.kind == K.ARRAY_SUBSCRIPT_EXPR:
        base, i = ca.children(n)
        idx.append(i)
        n = ca.strip(base)
    return idx


def features(sources, root=None, defines=(), includes=(), cplusplus=None):
    """Idiom features of the functions defined under root in the given source files."""
    weights = {k: 0.0 for k in ("affine", "indirect", "pointer", "other")}
    loops = {"counted": 0.0, "data": 0.0, "while": 0.0}
    state = {"max_depth": 0, "alloc_in_loop": 0, "references": 0, "errors": 0}
    seen = set()
    for source in sources:
        source = Path(source)
        cpp = cplusplus if cplusplus is not None else source.suffix in (".cc", ".cpp", ".cxx", ".hpp")
        tu = ca.parse(source, defines, includes, cplusplus=cpp)
        state["errors"] += sum(1 for d in tu.diagnostics if d.severity >= 3)
        project = Path(root) if root else source.parent

        def visit(n, depth, function=None, loop_vars=frozenset()):
            kind = n.kind
            if kind in (K.FUNCTION_DECL, K.CXX_METHOD, K.FUNCTION_TEMPLATE, K.CONSTRUCTOR, K.LAMBDA_EXPR):
                if kind != K.LAMBDA_EXPR:
                    if not n.is_definition() or not _in_project(n, project):
                        return
                    key = (str(n.location.file), n.location.offset)
                    if key in seen:
                        return
                    seen.add(key)
                    depth, function, loop_vars = 0, n, frozenset()
            if kind in LOOPS:
                weight = 10.0 ** min(depth, 4)
                own = set()
                if kind == K.FOR_STMT:
                    init, cond, _, _ = ca.for_parts(n)
                    own = _names(init) | ({c.spelling for c in init.get_children() if c.kind == K.VAR_DECL}
                                          if init is not None and init.kind == K.DECL_STMT else set())
                    data = cond is not None and _bound_from_data(cond, function, loop_vars, own)
                    loops["data" if data else "counted"] += weight
                elif kind == K.CXX_FOR_RANGE_STMT:
                    kids = ca.children(n)
                    own = {c.spelling for c in kids if c.kind == K.VAR_DECL}
                    ranges = [k for k in kids if k.kind.is_expression()]
                    loops["data" if ranges and _names(ranges[0]) & loop_vars else "counted"] += weight
                else:
                    loops["while"] += weight
                state["max_depth"] = max(state["max_depth"], depth + 1)
                inner = frozenset(loop_vars | own)
                for kid in n.get_children():
                    visit(kid, depth + 1, function, inner)
                return
            weight = 10.0 ** min(depth, 4)
            if kind == K.ARRAY_SUBSCRIPT_EXPR and not ca.is_array(n.type):
                weights[_worst([_index_class(i) for i in _subscript_chain(n)])] += weight
                state["references"] += 1
                # the bases and indices still hold references of their own
                for i in _subscript_chain(n):
                    visit(i, depth, function, loop_vars)
                return
            if kind == K.CALL_EXPR and n.spelling == "operator[]":
                kids = ca.children(n)
                weights["indirect" if _is_map(n) else _index_class(kids[-1]) if kids else "other"] += weight
                state["references"] += 1
            elif kind == K.CALL_EXPR and n.spelling in LOOKUPS and _is_map(n):
                weights["indirect"] += weight
                state["references"] += 1
            elif (kind == K.UNARY_OPERATOR and ca.unary_op(n) == "*") or \
                    (kind == K.CALL_EXPR and n.spelling in ("operator*", "operator->")):
                weights["pointer"] += weight
                state["references"] += 1
            elif kind == K.MEMBER_REF_EXPR and ca.children(n) and ca.is_pointer(ca.children(n)[0].type) \
                    and ca.strip(ca.children(n)[0]).kind != K.CXX_THIS_EXPR:
                weights["pointer"] += weight
                state["references"] += 1
            if kind == K.CALL_EXPR and depth > 0 and n.spelling in ALLOCATING:
                state["alloc_in_loop"] += 1
            if kind == K.CXX_NEW_EXPR and depth > 0:
                state["alloc_in_loop"] += 1
            for kid in n.get_children():
                visit(kid, depth, function, loop_vars)

        for c in tu.cursor.get_children():
            visit(c, 0)

    total = sum(weights.values()) or 1.0
    loop_total = sum(loops.values()) or 1.0
    out = {k: v / total for k, v in weights.items()}
    out.update({f"loops_{k}": v / loop_total for k, v in loops.items()})
    out.update({k: float(v) for k, v in state.items()})
    return out


def out_of_distribution(idiom, coverage=None, min_affine=0.95, max_data_loops=0.05, min_coverage=0.5):
    """Should a workload fall back to training-free sampling instead of the static spectrum or a
    borrowed model? Yes if the interpreter could not place most of its references, if they are
    not mostly affine, or if its loops are bounded by data."""
    if coverage is not None and (np.isnan(coverage) or coverage < min_coverage):
        return True
    return idiom["affine"] < min_affine or idiom["loops_data"] + idiom["loops_while"] > max_data_loops
