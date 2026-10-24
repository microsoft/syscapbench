"""Run untrusted cache replacement policies in a restricted child process.

Policy code produced by an LLM used to be written to ``cache/My.py`` and imported into the evaluator process, where it
could read the API keys loaded by ``main.py`` and run arbitrary commands (CWE-94). This module replaces that mechanism:

1. :func:`validate_policy_code` parses the code and rejects anything outside a small AST allow-list: imports of modules
   that are not allow-listed, dunder and private attribute access, frame/generator introspection, and so on.
2. :func:`run_policy_simulation` runs the simulation in a short-lived child interpreter (this file executed as a
   script) with an empty environment, restricted builtins, stand-ins for a few standard-library modules that expose
   only allow-listed members, and resource limits that the child cannot raise again (no new files, sockets or
   processes; bounded CPU time).

Policy code is never written to the source tree and never runs in the evaluator process. Anything a policy prints goes
to stderr. This is defense in depth rather than a hard security boundary, so untrusted evaluations should still run
inside a container.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import functools
import importlib
import json
import linecache
import os
import random
import re
import signal
import subprocess
import sys
import traceback
import types
from typing import Any, Callable

__all__ = [
    'POLICY_FUNCTIONS',
    'PolicySandboxError',
    'PolicyValidationError',
    'run_policy_simulation',
    'validate_policy_code',
]

POLICY_FUNCTIONS = ('evict', 'update_after_hit', 'update_after_insert', 'update_after_evict')

_SCRIPT_PATH = os.path.abspath(__file__)
_POLICY_FILENAME = '<policy>'
_RANDOM_SEED = 42

# Only these variables reach the child; everything else (API keys, tokens, ...) is dropped.
_FORWARDED_ENV_VARS = ('LD_LIBRARY_PATH', 'PYTHONHASHSEED', 'PYTHONHOME')

_MAX_ERROR_CHARS = 2_000
_MAX_TRACEBACK_CHARS = 20_000
_MAX_REPORTED_PROBLEMS = 20

# Module -> names that policy code may import from it (``None`` means every public name).
_IMPORTABLE: dict[str, tuple[str, ...] | None] = {
    'bisect': ('bisect', 'bisect_left', 'bisect_right', 'insort', 'insort_left', 'insort_right'),
    'collections': ('ChainMap', 'Counter', 'OrderedDict', 'defaultdict', 'deque', 'namedtuple'),
    'functools': ('cache', 'cached_property', 'cmp_to_key', 'lru_cache', 'partial', 'reduce', 'total_ordering'),
    'heapq': ('heapify', 'heappop', 'heappush', 'heappushpop', 'heapreplace', 'merge', 'nlargest', 'nsmallest'),
    'itertools': None,
    'logging': (
        'CRITICAL',
        'DEBUG',
        'ERROR',
        'FATAL',
        'INFO',
        'NOTSET',
        'WARN',
        'WARNING',
        'critical',
        'debug',
        'error',
        'exception',
        'fatal',
        'info',
        'log',
        'warning',
    ),
    'math': None,
    'operator': (
        'abs',
        'add',
        'and_',
        'concat',
        'contains',
        'countOf',
        'eq',
        'floordiv',
        'ge',
        'getitem',
        'gt',
        'index',
        'indexOf',
        'inv',
        'invert',
        'is_',
        'is_not',
        'itemgetter',
        'le',
        'length_hint',
        'lshift',
        'lt',
        'mod',
        'mul',
        'ne',
        'neg',
        'not_',
        'or_',
        'pos',
        'pow',
        'rshift',
        'sub',
        'truediv',
        'truth',
        'xor',
    ),
    'random': (
        'Random',
        'betavariate',
        'binomialvariate',
        'choice',
        'choices',
        'expovariate',
        'gammavariate',
        'gauss',
        'getrandbits',
        'getstate',
        'lognormvariate',
        'normalvariate',
        'paretovariate',
        'randbytes',
        'randint',
        'random',
        'randrange',
        'sample',
        'seed',
        'setstate',
        'shuffle',
        'triangular',
        'uniform',
        'vonmisesvariate',
        'weibullvariate',
    ),
    'statistics': (
        'StatisticsError',
        'correlation',
        'covariance',
        'fmean',
        'geometric_mean',
        'harmonic_mean',
        'linear_regression',
        'mean',
        'median',
        'median_grouped',
        'median_high',
        'median_low',
        'mode',
        'multimode',
        'pstdev',
        'pvariance',
        'quantiles',
        'stdev',
        'variance',
    ),
    'sys': ('float_info', 'getsizeof', 'maxsize'),
    'time': (
        'monotonic',
        'monotonic_ns',
        'perf_counter',
        'perf_counter_ns',
        'process_time',
        'process_time_ns',
        'time',
        'time_ns',
    ),
}

# ``typing`` is replaced by inert stubs: the real module offers ``get_type_hints``, which evaluates strings.
_TYPING_PLAIN_NAMES = (
    'AbstractSet',
    'Annotated',
    'Any',
    'Callable',
    'ClassVar',
    'Collection',
    'Container',
    'Final',
    'Generator',
    'Generic',
    'Hashable',
    'Iterable',
    'Iterator',
    'Literal',
    'Mapping',
    'MutableMapping',
    'MutableSequence',
    'MutableSet',
    'NoReturn',
    'Optional',
    'Protocol',
    'Reversible',
    'Sequence',
    'Sized',
    'Type',
    'Union',
)

# ``logging.getLogger()`` returns an inert stand-in: real loggers lead to their handlers' streams and file objects.
_LOGGER_METHODS = ('critical', 'debug', 'error', 'exception', 'fatal', 'info', 'log', 'warning')
_IGNORED_LOGGER_METHODS = ('addFilter', 'addHandler', 'removeFilter', 'removeHandler', 'setLevel')

# Notably absent: type, getattr (replaced by a checked version), setattr, delattr, vars, dir, globals, locals, eval,
# exec, compile, open, input, memoryview, breakpoint, help, exit and quit.
_SAFE_BUILTIN_NAMES = (
    'ArithmeticError',
    'AssertionError',
    'AttributeError',
    'BaseException',
    'Ellipsis',
    'Exception',
    'FloatingPointError',
    'ImportError',
    'IndexError',
    'KeyError',
    'LookupError',
    'MemoryError',
    'ModuleNotFoundError',
    'NameError',
    'NotImplemented',
    'NotImplementedError',
    'OverflowError',
    'RecursionError',
    'RuntimeError',
    'StopIteration',
    'TypeError',
    'UnboundLocalError',
    'ValueError',
    'ZeroDivisionError',
    'abs',
    'all',
    'any',
    'ascii',
    'bin',
    'bool',
    'bytearray',
    'bytes',
    'callable',
    'chr',
    'classmethod',
    'complex',
    'dict',
    'divmod',
    'enumerate',
    'filter',
    'float',
    'format',
    'frozenset',
    'hasattr',
    'hash',
    'hex',
    'id',
    'int',
    'isinstance',
    'issubclass',
    'iter',
    'len',
    'list',
    'map',
    'max',
    'min',
    'next',
    'object',
    'oct',
    'ord',
    'pow',
    'print',
    'property',
    'range',
    'repr',
    'reversed',
    'round',
    'set',
    'slice',
    'sorted',
    'staticmethod',
    'str',
    'sum',
    'super',
    'tuple',
    'zip',
)

_BINDABLE_DUNDERS = frozenset({'__all__', '__doc__', '__hash__', '__slots__'})
_READABLE_DUNDER_NAMES = frozenset({'__debug__', '__name__'})
_READABLE_DUNDER_ATTRIBUTES = frozenset({'__init__', '__name__'})
# Pattern matching reads the attributes listed in ``__match_args__`` with getattr().
_FORBIDDEN_CLASS_MEMBERS = frozenset({'__match_args__'})
_MANGLED_PRIVATE_NAME = re.compile(r'_[^_]\w*__\w')
_INTROSPECTION_ATTRIBUTES = frozenset(
    {
        'ag_await',
        'ag_code',
        'ag_frame',
        'ag_running',
        'cr_await',
        'cr_code',
        'cr_frame',
        'cr_origin',
        'cr_running',
        'cr_suspended',
        'f_back',
        'f_builtins',
        'f_code',
        'f_generator',
        'f_globals',
        'f_lasti',
        'f_lineno',
        'f_locals',
        'f_trace',
        'f_trace_lines',
        'f_trace_opcodes',
        'gi_code',
        'gi_frame',
        'gi_running',
        'gi_suspended',
        'gi_yieldfrom',
        'tb_frame',
        'tb_lasti',
        'tb_lineno',
        'tb_next',
    }
)


class PolicyValidationError(ValueError):
    """Raised when policy code uses a construct that is not allowed in the sandbox."""


class PolicySandboxError(RuntimeError):
    """Raised when a policy fails, crashes or times out inside the sandbox."""

    def __init__(self, message: str, child_traceback: str = '') -> None:
        """Create the error.

        Args:
            message: Short description of the failure.
            child_traceback: Traceback reported by the sandboxed process, if any.
        """
        super().__init__(message)
        self.child_traceback = child_traceback

    def __str__(self) -> str:
        """Return the message followed by the traceback reported by the sandbox, if any."""
        message = super().__str__()
        if not self.child_traceback:
            return message
        return f'{message}\n--- traceback from the policy sandbox ---\n{self.child_traceback}'


class _TypingStub:
    """Inert stand-in for a ``typing`` construct, so that type hints in policy code keep working."""

    __slots__ = ('_name', '_origin')

    def __init__(self, name: str, origin: type | None = None) -> None:
        self._name = name
        self._origin = origin

    def __getitem__(self, item: object) -> _TypingStub:
        return _TypingStub(self._name, self._origin)

    def __or__(self, other: object) -> _TypingStub:
        return _TypingStub('Union')

    __ror__ = __or__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._origin is None:
            raise TypeError(f'typing.{self._name} cannot be called in policy code')
        return self._origin(*args, **kwargs)

    def __mro_entries__(self, bases: tuple) -> tuple:
        return () if self._origin is None else (self._origin,)

    def __instancecheck__(self, obj: object) -> bool:
        if self._origin is None:
            raise TypeError(f'typing.{self._name} cannot be used with isinstance()')
        return isinstance(obj, self._origin)

    def __repr__(self) -> str:
        return f'typing.{self._name}'


def _type_var(name: str, *constraints: Any, **options: Any) -> _TypingStub:
    return _TypingStub(name)


def _cast(typ: Any, value: Any) -> Any:
    return value


def _typing_stubs() -> dict[str, Any]:
    import collections

    origins = {
        'Counter': collections.Counter,
        'DefaultDict': collections.defaultdict,
        'Deque': collections.deque,
        'Dict': dict,
        'FrozenSet': frozenset,
        'List': list,
        'OrderedDict': collections.OrderedDict,
        'Set': set,
        'Tuple': tuple,
    }
    stubs: dict[str, Any] = {name: _TypingStub(name) for name in _TYPING_PLAIN_NAMES}
    stubs.update({name: _TypingStub(name, origin) for name, origin in origins.items()})
    stubs.update({'TYPE_CHECKING': False, 'TypeVar': _type_var, 'cast': _cast})
    return stubs


def _ignore_logging_configuration(*args: Any, **kwargs: Any) -> None:
    """Accept and ignore logging configuration done by policy code."""


def _logging_stubs(functions: dict[str, Any]) -> dict[str, Any]:
    """Return stand-ins for ``logging.getLogger`` and ``logging.basicConfig``."""
    import logging

    logger = types.SimpleNamespace(
        isEnabledFor=lambda level: logging.getLogger().isEnabledFor(level),
        **{name: functions[name] for name in _LOGGER_METHODS},
        **dict.fromkeys(_IGNORED_LOGGER_METHODS, _ignore_logging_configuration),
    )

    def get_logger(name: str | None = None) -> types.SimpleNamespace:
        return logger

    return {'basicConfig': _ignore_logging_configuration, 'getLogger': get_logger}


@functools.cache
def _module_exports() -> dict[str, dict[str, Any]]:
    """Return, for each importable module, the objects that policy code may import from it."""
    exports: dict[str, dict[str, Any]] = {}
    for module_name, names in _IMPORTABLE.items():
        module = importlib.import_module(module_name)
        if names is None:
            names = tuple(name for name in dir(module) if not name.startswith('_'))
        exports[module_name] = {name: getattr(module, name) for name in names if hasattr(module, name)}
    future = importlib.import_module('__future__')
    exports['__future__'] = {name: getattr(future, name) for name in future.all_feature_names}
    exports['logging'].update(_logging_stubs(exports['logging']))
    exports['typing'] = _typing_stubs()
    return exports


def _attribute_problem(name: str, is_load: bool) -> str | None:
    """Return why accessing the attribute ``name`` is not allowed, or ``None`` if it is allowed."""
    if name.startswith('__'):
        if is_load and name in _READABLE_DUNDER_ATTRIBUTES:
            return None
        return f'access to the attribute {name!r} is not allowed'
    if _MANGLED_PRIVATE_NAME.match(name):
        return f'access to the private attribute {name!r} is not allowed'
    if name in _INTROSPECTION_ATTRIBUTES:
        return f'access to the introspection attribute {name!r} is not allowed'
    return None


def _checked_getattr(obj: object, name: str, *default: Any) -> Any:
    """Implement ``getattr`` for policy code with the same restrictions as the ``obj.name`` syntax."""
    if type(name) is not str:
        raise TypeError('getattr(): attribute name must be a str')
    problem = _attribute_problem(name, is_load=True)
    if problem is not None:
        raise AttributeError(f'{problem} in policy code')
    return getattr(obj, name, *default)


class _PolicyValidator:
    """Collect the constructs of a policy AST that are not allowed in the sandbox."""

    def __init__(self, tree: ast.Module) -> None:
        self.exports = _module_exports()
        self.problems: list[str] = []
        self.class_members = {
            id(statement) for node in ast.walk(tree) if isinstance(node, ast.ClassDef) for statement in node.body
        }

    def run(self, tree: ast.Module) -> list[str]:
        for node in ast.walk(tree):
            visit = getattr(self, f'_visit_{type(node).__name__}', None)
            if visit is not None:
                visit(node)
        return self.problems

    def _report(self, node: ast.AST, message: str) -> None:
        self.problems.append(f'line {getattr(node, "lineno", "?")}: {message}')

    def _check_binding(self, node: ast.AST, name: str) -> None:
        if name.startswith('__') and name not in _BINDABLE_DUNDERS:
            self._report(node, f'binding the name {name!r} is not allowed')

    def _check_attribute(self, node: ast.AST, name: str, is_load: bool) -> None:
        problem = _attribute_problem(name, is_load)
        if problem is not None:
            self._report(node, problem)

    def _allowed_modules(self) -> str:
        return ', '.join(sorted(name for name in self.exports if name != '__future__'))

    def _visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name not in self.exports:
                self._report(node, f'importing {alias.name!r} is not allowed (allowed: {self._allowed_modules()})')
            self._check_binding(node, alias.asname or alias.name.split('.')[0])

    def _visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module = node.module or ''
        if node.level:
            self._report(node, 'relative imports are not allowed')
        elif module not in self.exports:
            self._report(node, f'importing from {module!r} is not allowed (allowed: {self._allowed_modules()})')
        else:
            for alias in node.names:
                if alias.name != '*' and alias.name not in self.exports[module]:
                    self._report(node, f'importing {alias.name!r} from {module!r} is not allowed')
        for alias in node.names:
            if alias.name != '*':
                self._check_binding(node, alias.asname or alias.name)

    def _visit_Attribute(self, node: ast.Attribute) -> None:
        self._check_attribute(node, node.attr, isinstance(node.ctx, ast.Load))

    def _visit_Name(self, node: ast.Name) -> None:
        if not isinstance(node.ctx, ast.Load):
            self._check_binding(node, node.id)
        elif node.id.startswith('__') and node.id not in _READABLE_DUNDER_NAMES:
            self._report(node, f'use of the name {node.id!r} is not allowed')

    def _visit_arg(self, node: ast.arg) -> None:
        self._check_binding(node, node.arg)

    def _visit_Global(self, node: ast.Global | ast.Nonlocal) -> None:
        for name in node.names:
            self._check_binding(node, name)

    _visit_Nonlocal = _visit_Global

    def _visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self._check_binding(node, node.name)

    def _visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef) -> None:
        if not node.name.startswith('__'):
            return
        if id(node) not in self.class_members:
            self._report(node, f'defining {node.name!r} outside a class body is not allowed')
        elif node.name in _FORBIDDEN_CLASS_MEMBERS:
            self._report(node, f'defining {node.name!r} is not allowed')

    _visit_AsyncFunctionDef = _visit_FunctionDef
    _visit_ClassDef = _visit_FunctionDef

    def _visit_MatchAs(self, node: Any) -> None:
        if node.name:
            self._check_binding(node, node.name)

    _visit_MatchStar = _visit_MatchAs

    def _visit_MatchMapping(self, node: Any) -> None:
        if node.rest:
            self._check_binding(node, node.rest)

    def _visit_MatchClass(self, node: Any) -> None:
        for name in node.kwd_attrs:
            self._check_attribute(node, name, is_load=True)

    def _visit_TypeVar(self, node: Any) -> None:
        self._check_binding(node, node.name)

    _visit_ParamSpec = _visit_TypeVar
    _visit_TypeVarTuple = _visit_TypeVar


def validate_policy_code(code: str) -> ast.Module:
    """Check that policy code only uses constructs that are allowed in the sandbox.

    Args:
        code: Source code of the policy.

    Returns:
        The parsed module.

    Raises:
        PolicyValidationError: If the code cannot be parsed or uses a construct that is not allowed.
    """
    if not isinstance(code, str):
        raise PolicyValidationError(f'policy code must be a str, not {type(code).__name__}')
    try:
        tree = ast.parse(code, filename=_POLICY_FILENAME)
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError) as error:
        raise PolicyValidationError(f'policy code cannot be parsed: {error}') from None
    problems = _PolicyValidator(tree).run(tree)
    if problems:
        shown = problems[:_MAX_REPORTED_PROBLEMS]
        if len(problems) > len(shown):
            shown.append(f'... and {len(problems) - len(shown)} more')
        raise PolicyValidationError('policy code is not allowed in the sandbox:\n  ' + '\n  '.join(shown))
    return tree


def run_policy_simulation(
    code: str,
    config: Any,
    trace: Any,
    timeout: float | None = None,
    memory_limit: int | None = None,
) -> float:
    """Simulate a cache replacement policy in a sandboxed child process.

    Args:
        code: Source code of the policy. It must define the functions listed in ``POLICY_FUNCTIONS``.
        config: The cache configuration, e.g., a ``CacheConfig`` (anything whose ``to_dict()`` returns its arguments).
        trace: The requests to replay, as objects with ``key`` and ``size`` attributes (e.g., ``CacheObj``).
        timeout: Wall-clock limit for the child process in seconds. It also bounds the child's CPU time.
        memory_limit: Optional address-space limit for the child process in bytes.

    Returns:
        The miss ratio of the policy on the trace, rounded to 4 decimal places.

    Raises:
        PolicyValidationError: If the code uses a construct that is not allowed.
        PolicySandboxError: If the policy fails, crashes or times out.
    """
    validate_policy_code(code)
    if not sys.executable:
        raise PolicySandboxError('cannot locate the Python interpreter to run the policy sandbox')
    config_args = dict(config.to_dict())
    config_args['trace_path'] = os.path.abspath(config_args['trace_path'])
    keys, sizes = [], []
    for obj in trace:
        keys.append(obj.key)
        sizes.append(obj.size)
    request = {
        'code': code,
        'config': config_args,
        'keys': keys,
        'sizes': sizes,
        'cpu_limit': None if timeout is None else int(timeout) + 2,
        'memory_limit': memory_limit,
    }
    try:
        completed = subprocess.run(
            [sys.executable, '-S', '-B', _SCRIPT_PATH],
            input=json.dumps(request).encode('ascii'),
            stdout=subprocess.PIPE,
            env={name: os.environ[name] for name in _FORWARDED_ENV_VARS if name in os.environ},
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise PolicySandboxError(f'policy simulation timed out after {timeout} seconds') from None
    return _parse_response(completed.returncode, completed.stdout)


def _describe_exit(returncode: int) -> str:
    if returncode >= 0:
        return f'exited with status {returncode}'
    try:
        return f'was killed by {signal.Signals(-returncode).name}'
    except ValueError:
        return f'was killed by signal {-returncode}'


def _parse_response(returncode: int, stdout: bytes) -> float:
    try:
        response = json.loads(stdout.decode('ascii'))
    except ValueError:
        response = None
    if not isinstance(response, dict):
        raise PolicySandboxError(f'policy sandbox {_describe_exit(returncode)} without reporting a result')
    if response.get('ok') is not True:
        error = str(response.get('error') or 'unknown error')[:_MAX_ERROR_CHARS]
        child_traceback = str(response.get('traceback') or '')[-_MAX_TRACEBACK_CHARS:]
        raise PolicySandboxError(f'policy failed in the sandbox: {error}', child_traceback)
    if returncode != 0:
        raise PolicySandboxError(f'policy sandbox {_describe_exit(returncode)}')
    miss_ratio = response.get('miss_ratio')
    if isinstance(miss_ratio, bool) or not isinstance(miss_ratio, (int, float)) or not 0 <= miss_ratio <= 1:
        raise PolicySandboxError(f'policy sandbox returned an invalid miss ratio: {miss_ratio!r}')
    return float(miss_ratio)


# ---------------------------------------------------------------------------------------------------------------------
# Child process. Everything below runs in the sandboxed interpreter started by ``run_policy_simulation``.
# ---------------------------------------------------------------------------------------------------------------------


def _safe_builtins() -> dict[str, Any]:
    proxies = {name: types.SimpleNamespace(**members) for name, members in _module_exports().items()}

    def safe_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> Any:
        if level != 0:
            raise ImportError('relative imports are not allowed in policy code')
        # A SimpleNamespace has no __name__, so ``from x import y`` cannot fall back to sys.modules.
        proxy = proxies.get(name) if type(name) is str else None
        if proxy is None:
            raise ImportError(f'importing {name!r} is not allowed in policy code')
        return proxy

    safe = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES if hasattr(builtins, name)}
    safe.update(
        {
            '__build_class__': builtins.__build_class__,
            '__import__': safe_import,
            'getattr': _checked_getattr,
        }
    )
    return safe


def _compile_policy(code: str) -> types.CodeType:
    # Postponed evaluation keeps annotations such as ``obj: CacheObj`` from failing with a NameError.
    flags = importlib.import_module('__future__').annotations.compiler_flag
    return compile(validate_policy_code(code), _POLICY_FILENAME, 'exec', flags=flags, dont_inherit=True)


def _warm_up_tracebacks(code: str, source_files: tuple[str, ...]) -> None:
    """Load what traceback formatting needs while files can still be opened."""
    linecache.cache[_POLICY_FILENAME] = (len(code), None, code.splitlines(True), _POLICY_FILENAME)
    for path in source_files:
        linecache.getlines(path)
    for snippet in ('undefined_name', 'object().undefined_attribute', '{}[0]', '1 / 0'):
        try:
            exec(snippet, {'__builtins__': {'object': object}})  # noqa: S102
        except Exception:  # noqa: BLE001
            traceback.format_exc()


def _apply_resource_limits(cpu_limit: int | None, memory_limit: int | None) -> None:
    """Forbid new files, sockets and processes and bound CPU time and memory (best effort)."""
    try:
        import resource
    except ImportError:
        return
    limits = [('RLIMIT_CORE', 0), ('RLIMIT_NOFILE', 0), ('RLIMIT_NPROC', 0)]
    if cpu_limit:
        limits.append(('RLIMIT_CPU', int(cpu_limit)))
    if memory_limit:
        limits.append(('RLIMIT_AS', int(memory_limit)))
    for name, value in limits:
        which = getattr(resource, name, None)
        if which is None:
            continue
        try:
            hard = resource.getrlimit(which)[1]
            if hard != resource.RLIM_INFINITY:
                value = min(value, hard)
            # Setting the hard limit too makes the limit irreversible for the policy.
            resource.setrlimit(which, (value, value))
        except (OSError, ValueError):
            pass


def _policy_functions(namespace: dict[str, Any]) -> types.SimpleNamespace:
    missing = [name for name in POLICY_FUNCTIONS if not callable(namespace.get(name))]
    if missing:
        raise PolicyValidationError(f'policy code does not define the function(s): {", ".join(missing)}')
    return types.SimpleNamespace(**{name: namespace[name] for name in POLICY_FUNCTIONS})


def _simulate(request: dict[str, Any]) -> float:
    script_dir = os.path.dirname(_SCRIPT_PATH)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    cache_module = importlib.import_module('Cache')

    config = cache_module.CacheConfig(**request['config'])
    keys, sizes = request['keys'], request['sizes']
    if len(keys) != len(sizes):
        raise ValueError('trace keys and sizes differ in length')
    trace = [
        cache_module.CacheObj(key=key, size=size, consider_obj_size=config.consider_obj_size)
        for key, size in zip(keys, sizes)
    ]
    code = request['code']
    compiled = _compile_policy(code)
    namespace = {'__builtins__': _safe_builtins(), '__name__': 'policy'}
    _warm_up_tracebacks(code, (cache_module.__file__, _SCRIPT_PATH))
    _apply_resource_limits(request.get('cpu_limit'), request.get('memory_limit'))

    random.seed(_RANDOM_SEED)
    # From here on, untrusted code runs, but only in this restricted child process.
    exec(compiled, namespace)  # noqa: S102
    cache = cache_module.Cache(config, _policy_functions(namespace))
    if cache.access_count != 0 or cache.hit_count != 0:
        raise RuntimeError('a new cache must start with no accesses')
    for obj in trace:
        cache.get(obj)
    return round(1 - cache.hit_count / cache.access_count, 4)


def _bounded_text(produce: Callable[[], str], limit: int) -> str:
    """Call ``produce`` (which may run policy code) and keep the end of its result."""
    try:
        text = str.__str__(produce())
    except BaseException:  # noqa: BLE001
        return '<unavailable>'
    return text if len(text) <= limit else '...' + text[-limit:]


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view) :]


def _child_main() -> None:
    # Keep the real stdout for the result and send everything the policy prints to stderr instead.
    result_fd = os.dup(1)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    try:
        response = {'ok': True, 'miss_ratio': _simulate(json.loads(sys.stdin.buffer.read()))}
    except BaseException as error:  # noqa: BLE001 - also report SystemExit, KeyboardInterrupt, ...
        response = {
            'ok': False,
            'error': _bounded_text(functools.partial(repr, error), _MAX_ERROR_CHARS),
            'traceback': _bounded_text(traceback.format_exc, _MAX_TRACEBACK_CHARS),
        }
    _write_all(result_fd, json.dumps(response).encode('ascii'))
    os.close(result_fd)
    with contextlib.suppress(OSError, ValueError):
        sys.stderr.flush()
    # Skip interpreter shutdown so that policy finalizers and atexit hooks cannot hang or change the outcome.
    os._exit(0)


if __name__ == '__main__':
    _child_main()
