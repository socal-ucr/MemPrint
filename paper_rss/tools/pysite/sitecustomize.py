# option 3 for CPython: every MPR_INT seconds, RSS and pymalloc's free bytes (arena bytes - allocated blocks)
import os
if os.environ.get("MPR_PYOUT"):
    import sys, tempfile, threading, time
    def _loop(path=os.environ["MPR_PYOUT"], dt=float(os.environ.get("MPR_INT", "0.005"))):
        t0, page = time.monotonic(), os.sysconf("SC_PAGE_SIZE")
        tmp = tempfile.TemporaryFile()
        with open(path, "w") as out:
            out.write("seconds,rss_bytes,anon_bytes,free_bytes\n")
            while True:
                tmp.seek(0); tmp.truncate()
                sys.stderr.flush(); save = os.dup(2); os.dup2(tmp.fileno(), 2)
                try:
                    sys._debugmallocstats()
                finally:
                    os.dup2(save, 2); os.close(save)
                tmp.seek(0)
                s = {l.split(b"=")[0].strip(): int(l.split(b"=")[1].replace(b",", b"")) for l in tmp
                     if l.startswith(b"# ") and b"=" in l}
                free = s[b"# arenas allocated current"] * 256 * 1024 - s[b"# bytes in allocated blocks"]
                with open("/proc/self/statm") as f:
                    _, res, shared = map(int, f.read().split()[:3])
                out.write("%.4f,%d,%d,%d\n" % (time.monotonic() - t0, res * page, (res - shared) * page, free))
                out.flush()
                time.sleep(dt)
    threading.Thread(target=_loop, daemon=True).start()
