import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIN = (ROOT / "configs/llama.cpp.commit").read_text().strip()


@pytest.mark.parametrize("script", sorted((ROOT / 'scripts').rglob('*.sh')))
def test_shell_syntax(script):
    result = subprocess.run(['bash', '-n', str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def executable(path, body):
    path.write_text(f"#!{sys.executable}\n" + body)
    path.chmod(0o755)
    return path


@pytest.fixture
def benchmark(tmp_path):
    capture = tmp_path / "arguments.jsonl"
    cli = executable(tmp_path / "fake llama-cli", '''
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ['CAPTURE'], 'a') as stream:
    stream.write(json.dumps(args) + '\\n')
if os.environ.get('FAKE_FAIL'):
    print('simulated backend failure')
    sys.exit(7)
log = pathlib.Path(args[args.index('--log-file') + 1])
layers = '0' if args[args.index('-ngl') + 1] == '0' else '33'
if os.environ.get('FAKE_NO_OFFLOAD'): layers = '0'
text = f'load_tensors: offloaded {layers}/33 layers to GPU\\n'
text += '| prompt eval time = 1 ms (5.00 tokens per second)\\n'
if not os.environ.get('FAKE_INCOMPLETE'):
    text += '|        eval time = 1 ms (4.00 tokens per second)\\n'
log.write_text(text)
print(text)
''')
    model = tmp_path / "model with spaces.gguf"
    model.write_text("fixture, not a real model")
    env = os.environ.copy()
    for name in ["RUNS", "CONTEXT_SIZE", "PREDICT_TOKENS", "BATCH_SIZE", "MICROBATCH_SIZE", "FLASH_ATTENTION", "CPU_THREADS", "DEVICE"]:
        env.pop(name, None)
    env.update(LLAMA_CLI=str(cli), MODEL=str(model), RESULTS_DIR=str(tmp_path / "results"), CAPTURE=str(capture))

    def run(backend="gpu", **overrides):
        return subprocess.run(["bash", str(ROOT / f"scripts/benchmark-{backend}.sh")],
                              cwd=tmp_path, env=env | overrides, capture_output=True, text=True)

    return run, capture, tmp_path / "results"


@pytest.mark.parametrize("backend", ["cpu", "gpu"])
def test_arguments_metadata_and_three_runs(benchmark, backend):
    run, capture, results = benchmark
    result = run(backend, MICROBATCH_SIZE="32")
    assert result.returncode == 0, result.stderr
    commands = [json.loads(line) for line in capture.read_text().splitlines()]
    assert len(commands) == 3
    for args in commands:
        assert args[args.index('-ub') + 1] == '32'
        assert args[args.index('-b') + 1] == '32'
        assert args[args.index('-c') + 1] == '512'
        assert args[args.index('-n') + 1] == '128'
        assert args[args.index('-fa') + 1] == 'off'
        assert args[args.index('-ngl') + 1] == ('0' if backend == 'cpu' else '99')
        if backend == 'cpu':
            assert args[args.index('-t') + 1] == '6'
            assert '--device' not in args
        else:
            assert args[args.index('--device') + 1] == 'ROCm0'
    directory, = results.iterdir()
    metadata = (directory / 'metadata.txt').read_text()
    assert 'Microbatch size: 32' in metadata
    assert f'Pinned llama.cpp commit: {PIN}' in metadata
    assert 'Executable SHA256:' in metadata
    assert metadata.count(' command:') == 3
    assert 'Completed runs: 3/3' in (directory / 'status.txt').read_text()


def test_gpu_overrides_are_forwarded_and_recorded(benchmark):
    run, capture, results = benchmark
    result = run(RUNS='1', BATCH_SIZE='64', MICROBATCH_SIZE='16', CONTEXT_SIZE='1024',
                 PREDICT_TOKENS='16', DEVICE='ROCm1', FLASH_ATTENTION='on')
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())
    for flag, value in [('-b','64'),('-ub','16'),('-c','1024'),('-n','16'),('--device','ROCm1'),('-fa','on')]:
        assert args[args.index(flag) + 1] == value
    metadata = next(results.iterdir()).joinpath('metadata.txt').read_text()
    for value in ['Batch size: 64', 'Microbatch size: 16', 'Device: ROCm1', 'Flash Attention: on']:
        assert value in metadata


def test_cpu_thread_override_and_fixed_flash_attention(benchmark):
    run, capture, results = benchmark
    result = run('cpu', RUNS='1', CPU_THREADS='4', FLASH_ATTENTION='on')
    assert result.returncode == 0, result.stderr
    args = json.loads(capture.read_text())
    assert args[args.index('-t') + 1] == '4'
    assert args[args.index('-fa') + 1] == 'off'
    metadata = next(results.iterdir()).joinpath('metadata.txt').read_text()
    assert 'CPU threads: 4' in metadata
    assert 'Flash Attention: off' in metadata


@pytest.mark.parametrize("overrides", [
    {'RUNS':'0'}, {'BATCH_SIZE':'bad'}, {'MICROBATCH_SIZE':'33'},
    {'PREDICT_TOKENS':'-1'}, {'FLASH_ATTENTION':'invalid'}, {'DEVICE':'none'},
])
def test_invalid_inputs_do_not_invoke_inference(benchmark, overrides):
    run, capture, results = benchmark
    assert run(**overrides).returncode != 0
    assert not capture.exists()
    assert not results.exists()


@pytest.mark.parametrize("overrides,code", [({'FAKE_FAIL':'1'},7), ({'FAKE_INCOMPLETE':'1'},1), ({'FAKE_NO_OFFLOAD':'1'},1)])
def test_failures_keep_evidence_and_status(benchmark, overrides, code):
    run, _, results = benchmark
    result = run(**overrides)
    assert result.returncode == code
    directory, = results.iterdir()
    assert (directory / 'run-1.out').exists()
    assert (directory / 'metadata.txt').exists()
    status = (directory / 'status.txt').read_text()
    assert 'Status: failed' in status
    assert 'Completed runs: 0/3' in status


def test_new_invocations_do_not_overwrite_results(benchmark):
    run, _, results = benchmark
    assert run(RUNS='1').returncode == 0
    original = {p: p.read_bytes() for p in results.rglob('*') if p.is_file()}
    assert run(RUNS='1').returncode == 0
    assert len(list(results.iterdir())) == 2
    assert all(p.read_bytes() == data for p, data in original.items())


def test_build_uses_native_target_without_real_git_or_cmake(tmp_path):
    root = tmp_path / 'project'
    (root / 'scripts').mkdir(parents=True)
    (root / 'configs').mkdir()
    (root / 'misc/llama.cpp/.git').mkdir(parents=True)
    shutil.copy(ROOT / 'scripts/build-llama.sh', root / 'scripts/build-llama.sh')
    (root / 'configs/llama.cpp.commit').write_text(PIN + '\n')
    tools = tmp_path / 'tools'
    tools.mkdir()
    capture = tmp_path / 'cmake.jsonl'
    executable(tools / 'git', f'''
import os, sys
if 'status' in sys.argv: print(' M source.cpp' if os.environ.get('FAKE_DIRTY') else '', end='')
elif 'rev-parse' in sys.argv: print({PIN!r})
else: sys.exit('Unexpected Git mutation')
''')
    executable(tools / 'cmake', '''
import json, os, sys
with open(os.environ['CAPTURE'], 'a') as stream: stream.write(json.dumps(sys.argv[1:]) + '\\n')
''')
    env = os.environ.copy()
    for key in ['BUILD_DIR', 'BUILD_JOBS', 'AMDGPU_TARGETS', 'HIPCXX']:
        env.pop(key, None)
    env.update(PATH=str(tools) + os.pathsep + env['PATH'], CAPTURE=str(capture))
    command = ['bash', str(root / 'scripts/build-llama.sh')]
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    config, build = [json.loads(line) for line in capture.read_text().splitlines()]
    assert '-DCMAKE_HIP_ARCHITECTURES=gfx1032' in config
    assert '-DGGML_HIP_NO_VMM=ON' in config
    assert config[config.index('-B') + 1].endswith('/build-gfx1032-test')
    assert build[-2:] == ['--parallel', '2']
    before = capture.read_bytes()
    result = subprocess.run(command, env=env | {'FAKE_DIRTY':'1'}, capture_output=True, text=True)
    assert result.returncode != 0
    assert capture.read_bytes() == before
