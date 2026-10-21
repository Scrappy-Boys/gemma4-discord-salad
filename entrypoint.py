#!/usr/bin/env python3
"""
SaladCloud entrypoint: runs the whole Gemma 4 E2B fine-tune and survives node interruptions.

On every (re)start it:
  1. loads run state from your R2 bucket (state.json)
  2. checks this node has enough VRAM, otherwise asks Salad for a different node
  3. gets the converted dataset (builds it the first time, then re-downloads it from R2)
  4. restores the latest good checkpoint, if there is one
  5. trains, uploading each new checkpoint to R2 in the background
  6. uploads the final LoRA adapter, then stops its own container group so billing ends

Guard rails against runaway bills: failure counter, stall watchdog, wall-clock budget.
All settings come from environment variables (see .env.example).
"""
import collections
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.error
import urllib.request

import cloud

HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(f"[salad-run {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- config
class Cfg:
    def __init__(self, e=os.environ):
        g = lambda k, d="": e.get(k, d)
        # where things live
        self.workdir = g("WORKDIR", "/workspace")
        # SaladCloud (for stopping the container group when we are done)
        self.salad_key = g("SALAD_API_KEY")
        self.org = g("ORGANIZATION_NAME")
        self.project = g("PROJECT_NAME")
        self.group = g("CONTAINER_GROUP_NAME")
        self.salad_base = g("SALAD_API_BASE", "https://api.salad.com")
        self.imds_base = g("IMDS_BASE", "http://169.254.169.254")
        # data
        self.dataset = g("DATASET", "mookiezi/Discord-Dialogues")
        self.data_input = g("DATA_INPUT")  # local parquet instead of Hugging Face (testing)
        self.sample_rows = int(g("SAMPLE_ROWS", "105000"))
        self.eval_every = int(g("EVAL_EVERY", "200"))
        # training
        self.model = g("MODEL", "unsloth/gemma-4-E2B-it")
        self.max_train_rows = int(g("MAX_TRAIN_ROWS", "100000"))
        self.max_eval_rows = int(g("MAX_EVAL_ROWS", "500"))
        self.max_seq_len = int(g("MAX_SEQ_LEN", "2048"))
        self.batch_size = int(g("BATCH_SIZE", "4"))
        self.grad_accum = int(g("GRAD_ACCUM", "4"))
        self.epochs = g("EPOCHS", "1")
        self.max_steps = g("MAX_STEPS", "-1")
        self.lr = g("LR", "2e-4")
        self.rank = g("RANK", "16")
        self.save_steps = g("SAVE_STEPS", "250")
        self.eval_steps = g("EVAL_STEPS", "250")
        self.keep_ckpts = g("KEEP_CHECKPOINTS", "3")
        self.train_cmd = shlex.split(g("TRAIN_CMD")) if g("TRAIN_CMD") else None
        # guard rails
        self.min_vram_mb = int(g("MIN_VRAM_MB", "22000"))
        self.skip_preflight = g("SKIP_PREFLIGHT") == "1"
        self.stall_minutes = float(g("STALL_MINUTES", "45"))
        self.max_failures = int(g("MAX_FAILURES", "3"))
        self.max_wallclock_hours = float(g("MAX_WALLCLOCK_HOURS", "36"))
        self.poll_seconds = float(g("UPLOAD_POLL_SECONDS", "60"))
        self.quiet_seconds = float(g("CKPT_QUIET_SECONDS", "8"))
        self.shutdown_wait = float(g("SHUTDOWN_WAIT_SECONDS", "600"))


# --------------------------------------------------------------------------- salad calls
# Salad's API sits behind a web firewall that rejects Python's default user-agent with a 403,
# so always identify ourselves explicitly.
UA = "gemma4-salad/1.0 (+https://github.com/Scrappy-Boys/gemma4-discord-salad)"


def _http(url, method, headers, body=None, timeout=30):
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"User-Agent": UA, "Accept": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def _post(url, headers, body=None, timeout=30):
    return _http(url, "POST", headers, body, timeout)


def _describe(e):
    """Error text including the server's response body, which says why a request was refused."""
    if isinstance(e, urllib.error.HTTPError):
        try:
            return f"{e} | {e.read()[:300].decode('utf-8', 'replace')}"
        except Exception:
            pass
    return str(e)


def group_url(cfg):
    q = lambda x: urllib.parse.quote(x, safe="")
    return (f"{cfg.salad_base}/api/public/organizations/{q(cfg.org)}/projects/{q(cfg.project)}"
            f"/containers/{q(cfg.group)}")


def check_salad_access(cfg):
    """Before spending hours training, make sure we are allowed to stop this group afterwards."""
    if not (cfg.salad_key and cfg.org and cfg.project and cfg.group):
        return False
    try:
        _http(group_url(cfg), "GET", {"Salad-Api-Key": cfg.salad_key}, timeout=20)
        log("Salad API check passed (this container is allowed to stop its own group)")
        return True
    except urllib.error.HTTPError as e:
        if e.code in (401, 403, 404):
            log(f"SALAD API CHECK FAILED: {_describe(e)}")
            return False
        log(f"Salad API check inconclusive ({_describe(e)}); continuing")
        return True
    except Exception as e:
        log(f"Salad API check inconclusive ({e}); continuing")
        return True


def stop_group(cfg):
    """Ask SaladCloud to stop this container group (this is what ends billing)."""
    if not (cfg.salad_key and cfg.org and cfg.project and cfg.group):
        log("SALAD_API_KEY / ORGANIZATION_NAME / PROJECT_NAME / CONTAINER_GROUP_NAME not all set;"
            " cannot stop the container group automatically. STOP IT BY HAND in the Salad portal.")
        return False
    url = group_url(cfg) + "/stop"
    for attempt in range(5):
        try:
            status = _post(url, {"Salad-Api-Key": cfg.salad_key})
            log(f"stop request accepted (HTTP {status})")
            return True
        except Exception as e:
            log(f"stop request failed (attempt {attempt + 1}/5): {_describe(e)}")
            time.sleep(min(30, 3 * (attempt + 1)))
    return False


def reallocate(cfg, reason):
    """Ask SaladCloud to move this instance to a different node."""
    try:
        _post(f"{cfg.imds_base}/v1/reallocate",
              {"Content-Type": "application/json", "Metadata": "true"},
              json.dumps({"Reason": reason}).encode(), timeout=10)
        log(f"reallocation requested: {reason}")
    except Exception as e:
        log(f"reallocation request failed ({e})")


def shutdown(cfg, why, code=0):
    log(f"shutting down: {why}")
    if stop_group(cfg):
        time.sleep(cfg.shutdown_wait)  # Salad will stop us; don't exit and get restarted
    else:
        log("COULD NOT STOP THE CONTAINER GROUP. Sleeping so it doesn't restart-loop; stop it in the portal.")
        time.sleep(min(300, cfg.shutdown_wait))
    sys.exit(code)


# --------------------------------------------------------------------------- run state
class Run:
    def __init__(self, cfg, store):
        self.cfg, self.store = cfg, store
        self.lock = threading.Lock()
        self.state = None

    def load(self):
        for attempt in range(4):
            try:
                self.state = self.store.get_json("state.json") or {
                    "status": "running", "first_start": time.time(), "boots": 0,
                    "failures": 0, "data_ready": False, "latest_checkpoint": None,
                    "latest_step": 0, "last_error": None,
                }
                return
            except Exception as e:
                log(f"could not read state from R2 (attempt {attempt + 1}/4): {e}")
                time.sleep(5 * (attempt + 1))
        raise RuntimeError("cannot reach R2 / read state.json; check R2_* settings")

    def save(self, **changes):
        with self.lock:
            self.state.update(changes)
            self.state["updated_at"] = time.time()
            self.store.put_json("state.json", self.state)


# --------------------------------------------------------------------------- steps
def preflight(cfg):
    if cfg.skip_preflight:
        return
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=30)
        rows = [r.split(",") for r in out.strip().splitlines()]
        name, mem = rows[0][0].strip(), max(int(r[1]) for r in rows)
    except Exception as e:
        log(f"no usable GPU on this node ({e})")
        reallocate(cfg, "no usable GPU")
        time.sleep(min(300, cfg.shutdown_wait))
        sys.exit(1)
    log(f"GPU: {name}, {mem} MB")
    if mem < cfg.min_vram_mb:
        reallocate(cfg, f"only {mem} MB VRAM")
        time.sleep(min(300, cfg.shutdown_wait))
        sys.exit(1)


def prepare_data(run):
    cfg, store = run.cfg, run.store
    data_dir = os.path.join(cfg.workdir, "data")
    shutil.rmtree(data_dir, ignore_errors=True)
    if run.state["data_ready"]:
        n = store.download_dir("data", data_dir)
        if n == 0:
            raise RuntimeError("state says data is ready but none found in R2")
        log(f"downloaded converted dataset from R2 ({n} files)")
        return data_dir
    log("converting dataset (first run only)")
    cmd = [sys.executable, "-u", os.path.join(HERE, "convert_dataset.py"),
           "--out", data_dir, "--eval-every", str(cfg.eval_every)]
    if cfg.sample_rows:
        cmd += ["--sample-rows", str(cfg.sample_rows)]
    cmd += ["--input", cfg.data_input] if cfg.data_input else ["--dataset", cfg.dataset]
    subprocess.run(cmd, check=True)
    store.delete_dir("data")  # clear leftovers from any half-finished earlier upload
    n = store.upload_dir(data_dir, "data")
    run.save(data_ready=True)
    log(f"uploaded converted dataset to R2 ({n} files)")
    return data_dir


def restore_checkpoint(run, out_dir):
    store = run.store
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    name = run.state.get("latest_checkpoint")
    if not name:
        log("no checkpoint yet; starting from step 0")
        return False
    dest = os.path.join(out_dir, name)
    store.download_dir(f"ckpt/{name}", dest)
    if not os.path.isfile(os.path.join(dest, "trainer_state.json")):
        log(f"{name} in R2 is incomplete; starting from step 0")
        shutil.rmtree(dest, ignore_errors=True)
        run.save(latest_checkpoint=None, latest_step=0)
        return False
    log(f"restored {name}; will resume")
    return True


CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def newest_checkpoint(out_dir):
    best = None
    for name in os.listdir(out_dir):
        m, p = CKPT_RE.match(name), os.path.join(out_dir, name)
        if m and os.path.isfile(os.path.join(p, "trainer_state.json")):
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, name, p)
    return best


def dir_signature(path):
    sig = []
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            st = os.stat(fp)
            sig.append((os.path.relpath(fp, path), st.st_size, st.st_mtime_ns))
    return sorted(sig)


def newest_mtime(path):
    return max((s[2] for s in dir_signature(path)), default=0) / 1e9


def try_upload_checkpoint(run, out_dir):
    """Upload the newest finished checkpoint; only then move the 'latest' pointer."""
    best = newest_checkpoint(out_dir)
    if not best:
        return False
    step, name, path = best
    if step <= run.state["latest_step"]:
        return False
    try:
        if time.time() - newest_mtime(path) < run.cfg.quiet_seconds:
            return False  # still being written
        before = dir_signature(path)
        run.store.delete_dir(f"ckpt/{name}")
        n = run.store.upload_dir(path, f"ckpt/{name}")
        if dir_signature(path) != before:
            log(f"{name} changed while uploading; will retry")
            return False
        old = run.state["latest_checkpoint"]
        run.save(latest_checkpoint=name, latest_step=step)
        for d in run.store.list_subdirs("ckpt"):  # keep the new one and the one before it
            if d not in (name, old):
                run.store.delete_dir(f"ckpt/{d}")
        log(f"checkpoint {name} safe in R2 ({n} files)")
        return True
    except Exception as e:  # the folder may have been rotated away mid-upload; just retry later
        log(f"checkpoint upload skipped: {e}")
        return False


def uploader_loop(run, out_dir, stop_evt):
    while not stop_evt.is_set():
        try:
            try_upload_checkpoint(run, out_dir)
        except Exception as e:
            log(f"uploader error: {e}")
        stop_evt.wait(run.cfg.poll_seconds)


def train_command(cfg, data_dir, out_dir, resume):
    base = cfg.train_cmd or [sys.executable, "-u", os.path.join(HERE, "train_gemma4_e2b.py")]
    cmd = base + [
        "--data", data_dir, "--out", out_dir, "--model", cfg.model,
        "--max-train-rows", str(cfg.max_train_rows), "--max-eval-rows", str(cfg.max_eval_rows),
        "--max-seq-length", str(cfg.max_seq_len), "--batch-size", str(cfg.batch_size),
        "--grad-accum", str(cfg.grad_accum), "--epochs", str(cfg.epochs),
        "--max-steps", str(cfg.max_steps), "--lr", str(cfg.lr), "--rank", str(cfg.rank),
        "--save-steps", str(cfg.save_steps), "--eval-steps", str(cfg.eval_steps),
        "--keep-checkpoints", str(cfg.keep_ckpts),
    ]
    return cmd + (["--resume"] if resume else [])


def run_training(run, data_dir, out_dir, resume):
    cfg = run.cfg
    cmd = train_command(cfg, data_dir, out_dir, resume)
    log("starting training: " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=HERE)
    tail = collections.deque(maxlen=300)
    last_out = [time.time()]

    def reader():
        for line in proc.stdout:  # progress bars use \r, which also counts as a line here
            sys.stdout.write(line)
            sys.stdout.flush()
            tail.append(line.rstrip())
            last_out[0] = time.time()

    rt = threading.Thread(target=reader, daemon=True)
    rt.start()
    stop_up = threading.Event()
    ut = threading.Thread(target=uploader_loop, args=(run, out_dir, stop_up), daemon=True)
    ut.start()

    outcome = "exit"
    while proc.poll() is None:
        time.sleep(2)
        if time.time() - last_out[0] > cfg.stall_minutes * 60:
            outcome = "stalled"
        elif time.time() - run.state["first_start"] > cfg.max_wallclock_hours * 3600:
            outcome = "budget"
        if outcome != "exit":
            proc.terminate()
            try:
                proc.wait(30)
            except subprocess.TimeoutExpired:
                proc.kill()
            break
    proc.wait()
    rt.join(10)
    stop_up.set()
    ut.join(300)
    return outcome, proc.returncode, list(tail)


def upload_final(run, out_dir):
    final = os.path.join(out_dir, "final-lora")
    for attempt in range(5):
        try:
            run.store.delete_dir("final-lora")
            n = run.store.upload_dir(final, "final-lora")
            log(f"final LoRA adapter uploaded ({n} files)")
            return
        except Exception as e:
            log(f"final upload failed (attempt {attempt + 1}/5): {e}")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("could not upload the final adapter to R2")


def record_failure(run, tail, reason):
    cfg = run.cfg
    n = run.state["failures"] + 1
    log(f"FAILURE {n}/{cfg.max_failures}: {reason}")
    try:
        run.store.put_text(f"logs/failure-{n}.txt", reason + "\n\n" + "\n".join(tail))
    except Exception as e:
        log(f"could not upload failure log: {e}")
    run.save(failures=n, last_error=reason[:500])
    if n >= cfg.max_failures:
        run.save(status="failed")
        shutdown(cfg, f"giving up after {n} failures", code=1)
    sys.exit(1)  # container restarts and tries again


# --------------------------------------------------------------------------- main
def main():
    cfg = Cfg()
    log("booting")
    try:
        store = cloud.from_env()
        run = Run(cfg, store)
        run.load()
    except Exception as e:
        shutdown(cfg, f"startup problem: {e}", code=1)  # stop, or it would restart-loop forever

    run.save(boots=run.state["boots"] + 1)
    log(f"boot #{run.state['boots']}, status={run.state['status']}, "
        f"latest checkpoint={run.state['latest_checkpoint']}")

    if run.state["status"] == "done":
        shutdown(cfg, "training already finished")
    if run.state["status"] == "failed":
        shutdown(cfg, f"run marked failed: {run.state.get('last_error')}", code=1)
    if time.time() - run.state["first_start"] > cfg.max_wallclock_hours * 3600:
        run.save(status="failed", last_error="wall-clock budget exceeded")
        shutdown(cfg, "wall-clock budget exceeded", code=1)

    preflight(cfg)

    if not check_salad_access(cfg):
        log("NOT TRAINING: the Salad key or the organization/project/group names are wrong, so this "
            "container could not stop itself afterwards and would bill forever. Fix SALAD_API_KEY, "
            "ORGANIZATION_NAME, PROJECT_NAME, CONTAINER_GROUP_NAME in the Salad portal, then restart. "
            "STOP THE GROUP in the portal if you are not fixing it now.")
        time.sleep(min(600, cfg.shutdown_wait))
        sys.exit(1)

    out_dir = os.path.join(cfg.workdir, "outputs")
    tail = []
    try:
        data_dir = prepare_data(run)
        resume = restore_checkpoint(run, out_dir)
        outcome, code, tail = run_training(run, data_dir, out_dir, resume)
    except SystemExit:
        raise
    except Exception:
        record_failure(run, tail, "setup error:\n" + traceback.format_exc())

    if outcome == "stalled":
        try_upload_checkpoint(run, out_dir)
        log(f"training printed nothing for {cfg.stall_minutes} min; moving to a new node")
        reallocate(cfg, "training stalled")
        time.sleep(min(300, cfg.shutdown_wait))
        sys.exit(1)
    if outcome == "budget":
        try_upload_checkpoint(run, out_dir)
        run.save(status="failed", last_error="wall-clock budget exceeded")
        shutdown(cfg, "wall-clock budget exceeded", code=1)

    if code == 0 and os.path.isdir(os.path.join(out_dir, "final-lora")):
        try:
            upload_final(run, out_dir)
        except Exception:
            record_failure(run, tail, "final upload failed:\n" + traceback.format_exc())
        run.save(status="done")
        shutdown(cfg, "training finished and adapter saved")

    record_failure(run, tail, f"training process exited with code {code}")


if __name__ == "__main__":
    main()
