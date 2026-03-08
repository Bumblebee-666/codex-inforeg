from __future__ import annotations

import argparse
from datetime import datetime
import getpass
import os
from pathlib import Path
import shlex
import subprocess
import sys

import paramiko


HOST = "10.82.34.5"
PORT = 15654
USERNAME = "shenxiang"
REMOTE_PATH = "/data/Lab105/huangjiapeng/codex/new_InfoReg_CVPR2025"
MANIFEST_NAME = ".codex_sync_manifest"
REPO_ROOT = Path(__file__).resolve().parent
SAFE_DIRECTORY = str(REPO_ROOT)
ENSURED_REMOTE_DIRS = ("ckpt", "logs", "results")


def git_command(*args: str) -> list[str]:
    return ["git", "-c", f"safe.directory={SAFE_DIRECTORY}", *args]


def run_command(
    command: list[str], *, capture_output: bool = False, check: bool = True, text: bool = True
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=text,
        capture_output=capture_output,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr.decode() if isinstance(result.stderr, bytes) else result.stderr or "").strip()
        stdout = (result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout or "").strip()
        details = stderr or stdout or f"exit code {result.returncode}"
        raise RuntimeError(f"Command failed: {' '.join(command)}\n{details}")
    return result


def has_local_changes() -> bool:
    result = run_command(git_command("status", "--porcelain"), capture_output=True)
    return bool(result.stdout.strip())


def has_staged_changes() -> bool:
    result = subprocess.run(git_command("diff", "--cached", "--quiet"), cwd=REPO_ROOT)
    return result.returncode == 1


def sync_local_git(message: str) -> str:
    if has_local_changes():
        print("Staging local changes...")
        run_command(git_command("add", "-A"))
        if has_staged_changes():
            print("Creating commit...")
            run_command(git_command("commit", "-m", message))
        else:
            print("No committable changes after staging.")
    else:
        print("No local changes to commit.")

    print("Pushing to GitHub...")
    run_command(git_command("push"))
    result = run_command(git_command("rev-parse", "--short", "HEAD"), capture_output=True)
    return result.stdout.strip()


def build_snapshot() -> tuple[bytes, str]:
    archive = run_command(git_command("archive", "--format=tar", "HEAD"), capture_output=True, text=False)
    manifest = run_command(git_command("ls-tree", "-r", "--name-only", "HEAD"), capture_output=True)
    return archive.stdout, manifest.stdout


def remote_deploy_script(remote_tar_path: str, remote_manifest_path: str) -> str:
    ensure_dirs = " ".join(f'"$target/{name}"' for name in ENSURED_REMOTE_DIRS)
    return """
set -euo pipefail
target={target}
old_manifest="$target/{manifest_name}"
new_manifest={new_manifest}
new_archive={new_archive}
mkdir -p "$target"
python_bin="$(command -v python3 || command -v python)"
if [ -z "$python_bin" ]; then
  echo "python is required on the remote server" >&2
  exit 1
fi
"$python_bin" - "$target" "$old_manifest" "$new_manifest" <<'PY'
import os
import sys


target, old_manifest, new_manifest = sys.argv[1:4]


def read_manifest(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


old_files = set(read_manifest(old_manifest))
new_files = set(read_manifest(new_manifest))
removed_files = sorted(old_files - new_files, reverse=True)

for relative_path in removed_files:
    full_path = os.path.join(target, relative_path)
    if os.path.isfile(full_path) or os.path.islink(full_path):
        try:
            os.remove(full_path)
        except FileNotFoundError:
            pass

candidate_directories = sorted(
    {{os.path.dirname(path) for path in removed_files if os.path.dirname(path)}},
    key=len,
    reverse=True,
)

for relative_path in candidate_directories:
    full_path = os.path.join(target, relative_path)
    if os.path.isdir(full_path):
        try:
            os.rmdir(full_path)
        except OSError:
            pass
PY
tar -xf "$new_archive" -C "$target"
mkdir -p {ensure_dirs}
mv "$new_manifest" "$old_manifest"
rm -f "$new_archive"
echo "Remote deploy complete: $target"
""".strip().format(
        target=shlex.quote(REMOTE_PATH),
        manifest_name=MANIFEST_NAME,
        new_manifest=shlex.quote(remote_manifest_path),
        new_archive=shlex.quote(remote_tar_path),
        ensure_dirs=ensure_dirs,
    )


def deploy_snapshot(commit_sha: str, password: str) -> None:
    archive_bytes, manifest_text = build_snapshot()
    remote_tar_path = f"/tmp/codex_sync_{commit_sha}.tar"
    remote_manifest_path = f"/tmp/codex_sync_{commit_sha}.manifest"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("Connecting to server...")
    client.connect(HOST, port=PORT, username=USERNAME, password=password, timeout=20)

    try:
        with client.open_sftp() as sftp:
            with sftp.file(remote_tar_path, "wb") as remote_tar:
                remote_tar.write(archive_bytes)
            with sftp.file(remote_manifest_path, "w") as remote_manifest:
                remote_manifest.write(manifest_text)

        print("Deploying snapshot on server...")
        deploy_command = "/bin/bash -lc " + shlex.quote(remote_deploy_script(remote_tar_path, remote_manifest_path))
        _, stdout, stderr = client.exec_command(deploy_command)
        exit_status = stdout.channel.recv_exit_status()
        output = stdout.read().decode().strip()
        error = stderr.read().decode().strip()
        if output:
            print(output)
        if exit_status != 0:
            raise RuntimeError(error or "Remote deploy failed")
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Commit, push, and deploy the current project to the server.")
    parser.add_argument("message", nargs="?", default="", help="Git commit message.")
    parser.add_argument("--skip-deploy", action="store_true", help="Only commit and push; skip the server deploy step.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    commit_message = args.message or f"sync update {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    try:
        commit_sha = sync_local_git(commit_message)
        if args.skip_deploy:
            print(f"Push complete at {commit_sha}. Server deploy skipped.")
            return 0

        password = (
            os.environ.get("SYNC_SERVER_PASSWORD")
            or getpass.getpass(f"Password for {USERNAME}@{HOST}:{PORT}: ")
        )
        deploy_snapshot(commit_sha, password)
        print(f"Sync complete at commit {commit_sha}.")
        return 0
    except KeyboardInterrupt:
        print("Sync cancelled.")
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
