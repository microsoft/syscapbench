"""Tests for the sandbox that runs LLM-generated cache replacement policies (``cache/policy_sandbox.py``)."""

import importlib.util
import json
import os
import random
import re
import signal
import struct
import subprocess
import sys
import textwrap
import types
from pathlib import Path

import pytest

BENCH_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BENCH_DIR / 'src'
CACHE_DIR = SRC_DIR / 'cache_simulator' / 'cache'
ZIPF_TRACE = BENCH_DIR / 'data' / 'benchmark' / 'trace' / 'zipf' / 'alpha1.29_ndv1135_len12798.oracleGeneral.bin'

# Miss ratios on ZIPF_TRACE with a capacity of 9 objects, computed by the original in-process simulator.
RECORDED_MISS_RATIOS = {'fifo': 0.5929, 'lru': 0.5302, 'lfu': 0.4033, '214': 0.3935, '214-pq': 0.3935}
RECORDED_RANDOM_MISS_RATIO = 0.5902


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Loaded by path because importing the ``cache_simulator.cache`` package pulls in numpy.
sandbox = _load_module('cache_bench_policy_sandbox', CACHE_DIR / 'policy_sandbox.py')
cache_lib = _load_module('cache_bench_cache', CACHE_DIR / 'Cache.py')

RANDOM_POLICY = """
import random


def evict(cache_snapshot, obj):
    return random.choice(sorted(cache_snapshot.cache))


def update_after_hit(cache_snapshot, obj):
    pass


def update_after_insert(cache_snapshot, obj):
    pass


def update_after_evict(cache_snapshot, obj, evicted_obj):
    pass
"""

# LRU written with classes, typing hints, logging boilerplate and annotations that name undefined types (as LLMs often
# do).
TYPED_CLASS_LRU = """
import collections
import logging
import math
import sys
from typing import Dict, List, Optional

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
policy_name = __name__


class Recency:
    \"\"\"Remembers the order in which keys were used.\"\"\"

    __slots__ = ('order',)

    def __init__(self) -> None:
        self.order: Dict[str, int] = collections.OrderedDict()

    def __len__(self) -> int:
        return len(self.order)

    def __repr__(self) -> str:
        return f'Recency({len(self)})'


class LruTracker(Recency):
    __slots__ = ()

    def __init__(self) -> None:
        super().__init__()

    def touch(self, key: str, now: int) -> None:
        self.order[key] = now
        self.order.move_to_end(key)

    def oldest(self) -> Optional[str]:
        return next(iter(self.order), None)


tracker = LruTracker()
history: List[str] = []


def evict(cache_snapshot: CacheSnapshot, obj: CacheObj) -> str:
    victim = tracker.oldest()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug('evicting %s (%s, %r, %d bytes)', victim, math.inf, tracker, sys.getsizeof(history))
    return victim


def update_after_hit(cache_snapshot: CacheSnapshot, obj: CacheObj) -> None:
    tracker.touch(getattr(obj, 'key'), cache_snapshot.access_count)


def update_after_insert(cache_snapshot: CacheSnapshot, obj: CacheObj) -> None:
    tracker.touch(obj.key, cache_snapshot.access_count)


def update_after_evict(cache_snapshot: CacheSnapshot, obj: CacheObj, evicted_obj: CacheObj) -> None:
    del tracker.order[evicted_obj.key]
"""

# FIFO policy whose ``update_after_insert`` also runs the statements that replace ``BODY``.
FIFO_TEMPLATE = """
import collections

queue = collections.deque()


def evict(cache_snapshot, obj):
    return queue[0]


def update_after_hit(cache_snapshot, obj):
    pass


def update_after_insert(cache_snapshot, obj):
    queue.append(obj.key)
    BODY


def update_after_evict(cache_snapshot, obj, evicted_obj):
    queue.remove(evicted_obj.key)
"""


def fifo_policy(body='pass'):
    return FIFO_TEMPLATE.replace('    BODY', textwrap.indent(textwrap.dedent(body).strip(), '    '))


def read_sample(name):
    return (CACHE_DIR / 'sample_code' / f'{name}.py').read_text()


def make_config(trace_path=ZIPF_TRACE):
    return cache_lib.CacheConfig(
        capacity=9,
        consider_obj_size=False,
        trace_path=str(trace_path),
        key_col_id=1,
        size_col_id=2,
        has_header=False,
        delimiter=',',
    )


def simulate_in_process(code, config, trace):
    """Simulate a trusted policy the way the evaluator did before the sandbox existed."""
    state = random.getstate()
    random.seed(42)
    try:
        namespace = {'__name__': 'trusted_policy'}
        exec(compile(code, '<trusted policy>', 'exec'), namespace)  # noqa: S102 - trusted test fixtures only
        policy = types.SimpleNamespace(**{name: namespace[name] for name in sandbox.POLICY_FUNCTIONS})
        cache = cache_lib.Cache(config, policy)
        for obj in trace:
            cache.get(obj)
    finally:
        random.setstate(state)
    return round(1 - cache.hit_count / cache.access_count, 4)


@pytest.fixture(scope='module')
def synthetic_trace():
    rng = random.Random(2024)
    keys = rng.choices(range(200), weights=[1 / rank for rank in range(1, 201)], k=3000)
    return [cache_lib.CacheObj(key=str(key), size=1, consider_obj_size=False) for key in keys]


@pytest.fixture(scope='module')
def zipf_trace():
    # oracleGeneral records: timestamp, object id, object size, next access time.
    record = struct.Struct('<IQIq')
    return [
        cache_lib.CacheObj(key=str(key), size=1, consider_obj_size=False)
        for _, key, _, _ in record.iter_unpack(ZIPF_TRACE.read_bytes())
    ]


# ---------------------------------------------------------------------------------------------------------------------
# Static validation
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('name', sorted(RECORDED_MISS_RATIOS))
def test_sample_policies_pass_validation(name):
    sandbox.validate_policy_code(read_sample(name))


def test_common_policy_constructs_pass_validation():
    sandbox.validate_policy_code(
        textwrap.dedent(
            """
            from __future__ import annotations

            import bisect, functools, heapq, itertools, operator, statistics, sys, time
            from collections import Counter, OrderedDict, defaultdict, deque
            from typing import Any, Dict, List, Optional, Tuple

            __all__ = ['Entry']
            LIMIT = sys.maxsize


            class Entry:
                \"\"\"An entry.\"\"\"

                __slots__ = ('key', 'score')
                __hash__ = None

                def __init__(self, key: str, score: float) -> None:
                    super().__init__()
                    self.key = key
                    self.score = score

                def __lt__(self, other: Entry) -> bool:
                    return self.score < other.score


            def describe(obj: Any) -> str:
                return f'{__name__}: {getattr(obj, "key", None)!r} {obj.__init__.__name__}'
            """
        )
    )


REJECTED_CODE = [
    ('import os', "importing 'os' is not allowed"),
    ('import numpy as np', "importing 'numpy' is not allowed"),
    ('import copy', "importing 'copy' is not allowed"),
    ('import collections.abc', "importing 'collections.abc' is not allowed"),
    ('from collections import abc', "importing 'abc' from 'collections' is not allowed"),
    ('from os import system', "importing from 'os' is not allowed"),
    ('from . import Cache', 'relative imports are not allowed'),
    ('from operator import attrgetter', "importing 'attrgetter' from 'operator' is not allowed"),
    ('from functools import wraps', "importing 'wraps' from 'functools' is not allowed"),
    ("__import__('os')", "use of the name '__import__' is not allowed"),
    ('x = __builtins__', "use of the name '__builtins__' is not allowed"),
    ('__builtins__ = {}', "binding the name '__builtins__' is not allowed"),
    ('().__class__.__base__.__subclasses__()', "access to the attribute '__class__' is not allowed"),
    ('def f(obj):\n    return obj.__dict__', "access to the attribute '__dict__' is not allowed"),
    ("f'{obj.__class__}'", "access to the attribute '__class__' is not allowed"),
    ('obj.__init__ = None', "access to the attribute '__init__' is not allowed"),
    ('def g():\n    yield\nframe = g().gi_frame', "access to the introspection attribute 'gi_frame' is not allowed"),
    ('def f(frame):\n    return frame.f_back', "access to the introspection attribute 'f_back' is not allowed"),
    ('def f(cache):\n    return cache._Cache__nhit', "access to the private attribute '_Cache__nhit' is not allowed"),
    ("class C:\n    __match_args__ = ('__class__',)", "binding the name '__match_args__' is not allowed"),
    ('class C:\n    @property\n    def __match_args__(self):\n        return ()', "defining '__match_args__' is not"),
    ('class __builtins__:\n    pass', "defining '__builtins__' outside a class body is not allowed"),
    ('def __getattr__(name):\n    return name', "defining '__getattr__' outside a class body is not allowed"),
    ('def f():\n    global __builtins__', "binding the name '__builtins__' is not allowed"),
    ('try:\n    pass\nexcept Exception as __class__:\n    pass', "binding the name '__class__' is not allowed"),
    ('f = lambda __class__: 0', "binding the name '__class__' is not allowed"),
    ('def f(:', 'policy code cannot be parsed'),
]


@pytest.mark.parametrize(('code', 'expected'), REJECTED_CODE)
def test_unsafe_constructs_are_rejected(code, expected):
    with pytest.raises(sandbox.PolicyValidationError, match=re.escape(expected)):
        sandbox.validate_policy_code(code)


@pytest.mark.skipif(sys.version_info < (3, 10), reason='pattern matching requires Python 3.10')
@pytest.mark.parametrize(
    ('code', 'expected'),
    [
        ('match x:\n    case object(__class__=c):\n        pass', "access to the attribute '__class__' is not allowed"),
        ('match x:\n    case {**__builtins__}:\n        pass', "binding the name '__builtins__' is not allowed"),
    ],
)
def test_unsafe_patterns_are_rejected(code, expected):
    with pytest.raises(sandbox.PolicyValidationError, match=re.escape(expected)):
        sandbox.validate_policy_code(code)


def test_non_string_code_is_rejected():
    with pytest.raises(sandbox.PolicyValidationError, match='must be a str'):
        sandbox.validate_policy_code(b'import os')


def test_invalid_code_is_rejected_before_starting_the_sandbox(monkeypatch, synthetic_trace):
    def fail(*args, **kwargs):
        raise AssertionError('the sandbox must not be started for invalid code')

    monkeypatch.setattr(sandbox.subprocess, 'run', fail)
    with pytest.raises(sandbox.PolicyValidationError, match="importing 'os' is not allowed"):
        sandbox.run_policy_simulation('import os\n' + fifo_policy(), make_config(), synthetic_trace, timeout=60)


# ---------------------------------------------------------------------------------------------------------------------
# Simulation results
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize('name', [*sorted(RECORDED_MISS_RATIOS), 'random'])
def test_sandbox_matches_in_process_simulation(name, synthetic_trace):
    code = RANDOM_POLICY if name == 'random' else read_sample(name)
    expected = simulate_in_process(code, make_config(), synthetic_trace)
    assert sandbox.run_policy_simulation(code, make_config(), synthetic_trace, timeout=60) == expected


@pytest.mark.parametrize(('name', 'expected'), sorted(RECORDED_MISS_RATIOS.items()))
def test_sandbox_reproduces_recorded_miss_ratios(name, expected, zipf_trace):
    assert sandbox.run_policy_simulation(read_sample(name), make_config(), zipf_trace, timeout=120) == expected


def test_random_policy_is_seeded_like_before(zipf_trace):
    results = {sandbox.run_policy_simulation(RANDOM_POLICY, make_config(), zipf_trace, timeout=120) for _ in range(2)}
    assert results == {RECORDED_RANDOM_MISS_RATIO}


def test_common_llm_idioms_are_supported(synthetic_trace):
    expected = simulate_in_process(read_sample('lru'), make_config(), synthetic_trace)
    assert sandbox.run_policy_simulation(TYPED_CLASS_LRU, make_config(), synthetic_trace, timeout=60) == expected


def test_policy_output_cannot_spoof_the_result(capfd, synthetic_trace):
    trace = synthetic_trace[:200]
    fake_result = json.dumps({'ok': True, 'miss_ratio': 0.0})
    code = f'print({fake_result!r})\n' + fifo_policy(f'print({fake_result!r})')
    expected = simulate_in_process(fifo_policy(), make_config(), trace)
    assert expected > 0
    assert sandbox.run_policy_simulation(code, make_config(), trace, timeout=60) == expected
    assert fake_result in capfd.readouterr().err


def test_simulation_does_not_write_into_the_source_tree(synthetic_trace):
    before = sorted(os.listdir(CACHE_DIR))
    sandbox.run_policy_simulation(read_sample('lru'), make_config(), synthetic_trace[:200], timeout=60)
    assert sorted(os.listdir(CACHE_DIR)) == before
    assert not (CACHE_DIR / 'My.py').exists()


# ---------------------------------------------------------------------------------------------------------------------
# Containment of misbehaving policies
# ---------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('body', 'expected'),
    [
        ("open('escaped.txt', 'w')", "NameError(\"name 'open' is not defined"),
        ("eval('1 + 1')", "NameError(\"name 'eval' is not defined"),
        ('type(obj)', "NameError(\"name 'type' is not defined"),
        ("getattr(obj, '__class__')", "access to the attribute '__class__' is not allowed in policy code"),
        ("getattr(cache_snapshot, '_Cache__nhit')", "access to the private attribute '_Cache__nhit' is not allowed"),
        ("class Name(str):\n    pass\ngetattr(obj, Name('__class__'))", 'attribute name must be a str'),
        (
            "import logging\nlogging.warning('configures a handler')\nlogging.getLogger().handlers",
            "no attribute 'handlers'",
        ),
    ],
)
def test_dangerous_operations_fail_at_runtime(body, expected, synthetic_trace, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(sandbox.PolicySandboxError) as excinfo:
        sandbox.run_policy_simulation(fifo_policy(body), make_config(), synthetic_trace[:50], timeout=60)
    assert expected in str(excinfo.value)
    assert list(tmp_path.iterdir()) == []


def test_policy_cannot_read_the_evaluator_environment(monkeypatch, synthetic_trace):
    monkeypatch.setenv('CACHE_BENCH_FAKE_API_KEY', 'sk-cache-bench-fake-secret')
    # str.format() can still walk attributes, but it only produces strings and the sandbox environment is empty.
    body = "raise RuntimeError('ENV={0.get.__func__.__globals__[os].environ}'.format(cache_snapshot))"
    with pytest.raises(sandbox.PolicySandboxError) as excinfo:
        sandbox.run_policy_simulation(fifo_policy(body), make_config(), synthetic_trace[:50], timeout=60)
    message = str(excinfo.value)
    assert 'ENV=environ(' in message
    assert 'sk-cache-bench-fake-secret' not in message
    assert 'CACHE_BENCH_FAKE_API_KEY' not in message


def test_policy_exception_is_reported_with_its_traceback(synthetic_trace):
    with pytest.raises(sandbox.PolicySandboxError) as excinfo:
        sandbox.run_policy_simulation(
            fifo_policy("raise ValueError('policy bug')"), make_config(), synthetic_trace[:50], timeout=60
        )
    assert "ValueError('policy bug')" in str(excinfo.value)
    assert 'File "<policy>"' in excinfo.value.child_traceback
    assert "raise ValueError('policy bug')" in excinfo.value.child_traceback


def test_missing_policy_function_is_reported(synthetic_trace):
    code = fifo_policy().replace('def update_after_evict', 'def forget')
    with pytest.raises(sandbox.PolicySandboxError, match=r'does not define the function\(s\): update_after_evict'):
        sandbox.run_policy_simulation(code, make_config(), synthetic_trace[:50], timeout=60)


def test_infinite_loop_times_out(synthetic_trace):
    with pytest.raises(sandbox.PolicySandboxError, match='timed out after 1 seconds'):
        sandbox.run_policy_simulation(
            fifo_policy('while True:\n    pass'), make_config(), synthetic_trace[:50], timeout=1
        )


@pytest.mark.parametrize(
    ('returncode', 'stdout', 'expected'),
    [
        (-int(signal.SIGTERM), b'', 'policy sandbox was killed by SIGTERM without reporting a result'),
        (1, b'not json', 'policy sandbox exited with status 1 without reporting a result'),
        (0, b'[0.5]', 'policy sandbox exited with status 0 without reporting a result'),
        (0, b'\xff', 'policy sandbox exited with status 0 without reporting a result'),
        (0, b'{"ok": false, "error": "boom"}', 'policy failed in the sandbox: boom'),
        (3, b'{"ok": true, "miss_ratio": 0.5}', 'policy sandbox exited with status 3'),
        (0, b'{"ok": true, "miss_ratio": 1.5}', 'invalid miss ratio: 1.5'),
        (0, b'{"ok": true, "miss_ratio": true}', 'invalid miss ratio: True'),
        (0, b'{"ok": true, "miss_ratio": "0.5"}', "invalid miss ratio: '0.5'"),
    ],
)
def test_unexpected_sandbox_responses_are_rejected(returncode, stdout, expected):
    with pytest.raises(sandbox.PolicySandboxError, match=re.escape(expected)):
        sandbox._parse_response(returncode, stdout)


def test_valid_sandbox_response_is_accepted():
    assert sandbox._parse_response(0, b'{"ok": true, "miss_ratio": 0.25}') == 0.25


RESOURCE_LIMITS_PROBE = """
import importlib.util, json, os, resource, socket, sys

spec = importlib.util.spec_from_file_location('policy_sandbox', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module._apply_resource_limits(7, None)


def blocked(action):
    try:
        action()
    except (OSError, ValueError):
        return True
    return False


def fork():
    if os.fork() == 0:
        os._exit(0)
    os.wait()


names = ('RLIMIT_CORE', 'RLIMIT_CPU', 'RLIMIT_NOFILE', 'RLIMIT_NPROC')
print(json.dumps({
    'limits': {name: resource.getrlimit(getattr(resource, name)) for name in names},
    'open': blocked(lambda: open(sys.argv[1])),
    'pipe': blocked(os.pipe),
    'socket': blocked(socket.socket),
    'raise_nofile': blocked(lambda: resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))),
    'raise_cpu': blocked(lambda: resource.setrlimit(resource.RLIMIT_CPU, (100, 100))),
    'fork': blocked(fork),
}))
"""


def test_resource_limits_are_irreversible():
    pytest.importorskip('resource')
    completed = subprocess.run(
        [sys.executable, '-S', '-B', '-c', RESOURCE_LIMITS_PROBE, str(CACHE_DIR / 'policy_sandbox.py')],
        stdout=subprocess.PIPE,
        env={name: os.environ[name] for name in sandbox._FORWARDED_ENV_VARS if name in os.environ},
        timeout=60,
        check=True,
    )
    result = json.loads(completed.stdout)
    assert result['limits']['RLIMIT_CORE'] == [0, 0]
    assert result['limits']['RLIMIT_NOFILE'] == [0, 0]
    assert result['limits']['RLIMIT_NPROC'] == [0, 0]
    cpu_soft, cpu_hard = result['limits']['RLIMIT_CPU']
    assert cpu_soft == cpu_hard <= 7
    assert result['open'] and result['pipe'] and result['socket']
    assert result['raise_nofile'] and result['raise_cpu']
    if os.geteuid() != 0:  # root is exempt from RLIMIT_NPROC
        assert result['fork']


# ---------------------------------------------------------------------------------------------------------------------
# Integration with the simulator
# ---------------------------------------------------------------------------------------------------------------------


def test_simulator_runs_policies_in_the_sandbox(monkeypatch):
    pytest.importorskip('numpy')
    pytest.importorskip('matplotlib')
    monkeypatch.syspath_prepend(str(SRC_DIR))
    from cache_simulator.run_simulatorcache import simulate

    assert simulate(read_sample('lru'), trace_path=str(ZIPF_TRACE))[1] == RECORDED_MISS_RATIOS['lru']
    assert simulate('import os\n' + read_sample('lru'), trace_path=str(ZIPF_TRACE))[1] is None
    assert not (CACHE_DIR / 'My.py').exists()
