import os
import subprocess
from copy import deepcopy

from tempfile import NamedTemporaryFile, TemporaryDirectory
CLI_ARG_SIZE_LIMIT = 1024 * 3


_ERROR_MSG_PREFIX = "Failed to execute program: "
_DEFAULT_TIMEOUT_SECONDS = 35



def code_exec_firejail(code, stdin: str = None, timeout=_DEFAULT_TIMEOUT_SECONDS, pytest: str = None):
    env = {}
    for k, v in os.environ.copy().items():
        if "KML_" != k[:4]:
            env[k] = v
 
    env["OPENBLAS_NUM_THREADS"] = "1"
    if "PYTHONPATH" in env:
        del env["PYTHONPATH"]


    command = [
        "firejail",
        "--private",
        "--quiet",
        "--seccomp=socket",
        "--profile=pip",
        "--rlimit-nproc=32",
        "--rlimit-nofile=32",
        "--rlimit-fsize=8m",
        "--rlimit-as=1024m",
        f"--timeout=00:00:{timeout}",
        "--kill-at-exit",  
    ]

    
    if pytest:
        # solution is in {tmpdir}/solution.py
        with TemporaryDirectory() as tmpdir:
            assert stdin is None, "STDIN is not supported with pytest"
            # Write the solution to a file
            with open(os.path.join(tmpdir, "solution.py"), "w") as f:
                f.write(code)
            with open(os.path.join(tmpdir, "test_solution.py"), "w") as f:
                f.write(pytest)
            command.insert(4, f"--whitelist={tmpdir}")
            command.extend(["/share/miniconda3/envs/verl_v0_szp/bin/python", "-m", "pytest", tmpdir])
            try:
                result = subprocess.run(
                    command,
                    cwd=tmpdir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    check=False,
                    timeout=timeout + 3,
                )
            except subprocess.TimeoutExpired:
                return False, _ERROR_MSG_PREFIX + "执行超时，进程被强制终止"
    else:
        with NamedTemporaryFile() as tmp:
            tmp.write(code.encode())
            tmp.flush()
            command.insert(4, f"--whitelist={tmp.name}")
            command.extend(["/share/miniconda3/envs/verl_v0_szp/bin/python", tmp.name])
            try:
                result = subprocess.run(command,
                                    input=stdin.encode() if stdin else None,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    env=env,
                                    check=False,
                                    timeout=timeout)
            except subprocess.TimeoutExpired:
                print("Timeout")
                return False, _ERROR_MSG_PREFIX + "执行超时，进程被强制终止"

    stderr = result.stderr.decode().strip()
    stdout = result.stdout.decode()

    if result.returncode == 0:
        return True, stdout
    return False, _ERROR_MSG_PREFIX + f"STDOUT:\n{stdout}\n\nSTDERR:\n{stderr}"
