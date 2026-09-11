"""Remove orphaned POSIX shared-memory segments left in /dev/shm by killed vLLM processes.

vLLM's engine/worker processes create /dev/shm/psm_* segments (multiprocessing.shared_memory) and an
object-storage buffer; a watchdog kill never unlinks them and with --ipc=host they outlive the container
(133 leftovers = 690 MiB on the head, 2026-09-10). Runs as root in a throwaway container of the kit image
with --pid=host --ipc=host --cap-add SYS_PTRACE, and unlinks only segments no host process maps or holds open.
"""
import glob
import os

names = [os.path.basename(p) for p in glob.glob("/dev/shm/psm_*") + glob.glob("/dev/shm/VLLM_OBJECT_STORAGE_SHM_BUFFER_*")]
held: set[str] = set()
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        txt = open(f"/proc/{pid}/maps").read()
        held.update(n for n in names if n in txt)
        for fd in os.listdir(f"/proc/{pid}/fd"):
            try:
                link = os.readlink(f"/proc/{pid}/fd/{fd}")
            except OSError:
                continue
            held.update(n for n in names if n in link)
    except OSError:
        pass
freed = removed = 0
for n in names:
    if n in held:
        continue
    path = "/dev/shm/" + n
    try:
        freed += os.stat(path).st_blocks * 512
        os.unlink(path)
        removed += 1
    except OSError:
        pass
print(f"removed {removed} orphaned shm segments ({freed // 1048576} MiB), kept {len(held)} in use")
