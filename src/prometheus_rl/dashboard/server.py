"""Local-only browser UI for training, playback and experiment inspection."""

import argparse
import csv
import json
import os
import secrets
import subprocess
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from math import isfinite
from pathlib import Path
from urllib.parse import parse_qs, quote, urlsplit
from uuid import uuid4

from prometheus_rl.dashboard.commands import (
    ALGORITHMS,
    play_command,
    train_command,
)
from prometheus_rl.dashboard.results import aggregate, discover

STYLE = """
body{font:16px system-ui;background:#101827;color:#e5edf7;margin:32px auto;
max-width:1100px;padding:0 16px} h1{color:#78e0d2} section{background:#1b293b;
padding:20px;margin:20px 0;border-radius:12px} input,select,button{padding:8px;
margin:5px;border-radius:5px} button{background:#78e0d2;color:#102030;cursor:pointer}
table{border-collapse:collapse;width:100%} td,th{padding:8px;border-bottom:1px
solid #405065;text-align:left} a{color:#78e0d2} pre{white-space:pre-wrap;
max-height:340px;overflow:auto} small{color:#b5c4d6} svg{width:100%;height:auto}
"""


def chart(path: Path, column: str) -> str:
    try:
        with path.open(newline="") as handle:
            points = [(float(r["step"]), float(r[column]))
                      for r in csv.DictReader(handle) if r.get(column)]
    except (OSError, KeyError, ValueError):
        return "<p>No data yet.</p>"
    points = [(x, y) for x, y in points if isfinite(x) and isfinite(y)]
    if not points:
        return "<p>No data yet.</p>"
    lo, hi = min(y for _, y in points), max(y for _, y in points)
    xmax = max(x for x, _ in points) or 1
    stride = max(1, len(points) // 1000)
    coords = " ".join(f"{40+700*x/xmax:.2f},{160-130*(y-lo)/(hi-lo or 1):.2f}"
                      for x, y in points[::stride])
    return (f'<svg viewBox="0 0 780 190" role="img" aria-label="{escape(column)}">'
            '<path d="M40 15 V160 H740" fill="none" stroke="#a0b0bf"/>'
            f'<polyline points="{coords}" fill="none" stroke="#78e0d2"/>'
            f'<text x="42" y="12" fill="white">max {hi:.3g}</text>'
            f'<text x="42" y="180" fill="white">min {lo:.3g}</text>'
            f'<text x="620" y="180" fill="white">step {xmax:.0f}</text></svg>')


class Dashboard:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs = self.root / ".dashboard-logs"
        self.logs.mkdir(exist_ok=True)
        self.token = secrets.token_urlsafe(32)
        self.process = None
        self.log = None
        self.job = "No job started"

    def start(self, command: list[str], threads: int) -> None:
        if self.process is not None and self.process.poll() is None:
            raise ValueError("another job is running")
        self.log = self.logs / f"{uuid4().hex}.log"
        env = {**os.environ, "OMP_NUM_THREADS": str(threads),
               "MKL_NUM_THREADS": str(threads), "PYTHONUNBUFFERED": "1"}
        with self.log.open("w") as handle:
            self.process = subprocess.Popen(command, stdout=handle,
                                            stderr=subprocess.STDOUT, env=env)
        self.job = " ".join(command[3:])

    def submit(self, form: dict) -> None:
        if not secrets.compare_digest(form.get("token", ""), self.token):
            raise ValueError("invalid form token")
        if form.get("action") == "stop":
            if self.process is not None and self.process.poll() is None:
                self.process.terminate()
            return
        if form.get("action") == "play":
            runs, _ = discover(self.root)
            run = next((r for r in runs if r.path.name == form.get("run")), None)
            if run is None or not (run.path / "model.pt").is_file():
                raise ValueError("checkpoint unavailable")
            game = {"CartPole-v1": "cartpole", "Pendulum-v1": "pendulum",
                    "ALE/Breakout-v5": "breakout", "ALE/Pong-v5": "pong",
                    "Prometheus-snake-v0": "snake",
                    "Prometheus-gridworld-v0": "gridworld"}[run.environment]
            self.start(play_command(run.algorithm, game, run.path / "model.pt"), 1)
            return
        if form.get("action") != "train":
            raise ValueError("unknown action")
        algorithm, game = form["algorithm"], form["game"]
        threads = int(form["threads"])
        # Generate a new directory name, so prior results cannot be overwritten.
        output = self.root / f"run-{uuid4().hex[:12]}"
        cmd = train_command(algorithm, game, output, steps=int(form["steps"]),
                            seed=int(form["seed"]), episodes=int(form["episodes"]),
                            threads=threads, device=form["device"])
        self.start(cmd, threads)

    def job_info(self) -> dict:
        status = ("idle" if self.process is None else "running" if
                  self.process.poll() is None else f"exit {self.process.returncode}")
        log = ""
        if self.log is not None:
            with self.log.open("rb") as handle:
                handle.seek(max(0, self.log.stat().st_size - 12000))
                log = handle.read(12000).decode(errors="replace")
        return {"status": status, "log": log, "command": self.job}

    def page(self, message="", selected="") -> str:
        runs, errors = discover(self.root)
        token = f'<input type="hidden" name="token" value="{self.token}">'
        job = self.job_info()
        status, log = job["status"], job["log"]
        options = "".join(f"<option>{a}</option>" for a in ALGORITHMS)
        games = "".join(f"<option>{g}</option>" for g in
                        ("cartpole", "snake", "gridworld", "pendulum",
                         "breakout", "pong"))
        rows = []
        for r in runs:
            name = escape(r.path.name, quote=True)
            link = quote(r.path.name, safe="")
            play = (f'<form method="post">{token}<input type="hidden" name="run" '
                    f'value="{name}"><button name="action" value="play">Play'
                    '</button></form>')
            rows.append(f'<tr><td><a href="/?run={link}">{name}</a></td>'
                        f'<td>{escape(r.environment)}</td><td>{r.algorithm}</td>'
                        f'<td>{r.seed}</td><td>{r.score:.3f}</td>'
                        f'<td>{r.baseline:.3f}</td><td>{play}</td></tr>')
        detail = ""
        run = next((r for r in runs if r.path.name == selected), None)
        if run is not None:
            detail = (f'<section><h2>{escape(selected)} | episode returns</h2>'
                      + chart(run.path / "episodes.csv", "reward")
                      + '<h3>Loss / policy loss</h3>'
                      + chart(run.path / "episodes.csv", "mean_loss")
                      + chart(run.path / "updates.csv", "policy_loss")
                      + '</section>')
        return f'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Prometheus-RL</title><style>{STYLE}</style><body>
<h1>Prometheus-RL / Research Lab</h1><p>Local experiment console · v2.0.0</p>
<p>{escape(message)}</p><section><h2>Launch experiment</h2>
<form method="post">{token}<label>Algorithm <select name="algorithm">{options}
</select></label><label>Environment <select name="game">{games}</select></label>
<label>Steps <input name="steps" type="number" min="1" value="5000"></label>
<label>Seed <input name="seed" type="number" min="0" value="0"></label>
<label>Evaluation episodes <input name="episodes" type="number" value="3"></label>
<label>Threads <input name="threads" type="number" value="2"></label>
<label>Device <select name="device"><option>cpu</option><option>cuda</option>
</select></label><button name="action" value="train">Start training</button></form>
<small>CPU pilot: 5,000 steps. Atari uses reduced replay and evaluation budgets.
Choose a compatible algorithm and environment. Only one job runs at once.</small>
</section><section><h2>Job: <span id="status">{escape(status)}</span></h2>
<p id="command">{escape(self.job)}</p>
<form method="post">{token}<button name="action" value="stop">Stop job</button>
<a href="/">Refresh log</a></form><pre id="log">{escape(log)}</pre></section>
<section><h2>Completed experiments</h2><table><tr><th>Run</th><th>Environment</th>
<th>Algorithm</th><th>Seed</th><th>Mean</th><th>Random</th><th>Model</th></tr>
{''.join(rows)}</table><pre>{escape(chr(10).join(errors))}</pre></section>{detail}
<section><h2>Benchmark groups</h2><a href="/report.json">Export JSON report</a>
<pre>{escape(json.dumps(aggregate(runs), indent=2))}</pre></section>
<script>
setInterval(async () => {{
  try {{
    const response = await fetch('/job.json', {{cache: 'no-store'}});
    if (!response.ok) return;
    const job = await response.json();
    document.getElementById('status').textContent = job.status;
    document.getElementById('command').textContent = job.command;
    document.getElementById('log').textContent = job.log;
  }} catch {{ /* The user can refresh if the server disconnects. */ }}
}}, 2000);
</script></body></html>'''


def serve(root: Path, port: int) -> None:
    app = Dashboard(root)

    class Handler(BaseHTTPRequestHandler):
        def send_body(self, body, kind="text/html", status=200):
            payload = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", kind + "; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.end_headers()
            self.wfile.write(payload)

        def valid_host(self):
            return self.headers.get("Host") == f"127.0.0.1:{port}"

        def do_GET(self):
            if not self.valid_host():
                self.send_error(403)
                return
            url = urlsplit(self.path)
            if url.path == "/job.json":
                self.send_body(json.dumps(app.job_info()), "application/json")
            elif url.path == "/report.json":
                runs, errors = discover(app.root)
                self.send_body(json.dumps({"groups": aggregate(runs),
                                           "errors": errors}, indent=2),
                               "application/json")
            elif url.path == "/":
                self.send_body(app.page(selected=parse_qs(url.query).get(
                    "run", [""])[0]))
            else:
                self.send_error(404)

        def do_POST(self):
            if not self.valid_host() or self.path != "/":
                self.send_error(403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8192:
                    raise ValueError("invalid request size")
                form = {k: v[0] for k, v in parse_qs(
                    self.rfile.read(size).decode()).items()}
                app.submit(form)
            except (ValueError, KeyError, OSError) as error:
                self.send_body(app.page(str(error)), status=400)
                return
            self.send_response(303)
            self.send_header("Location", "/")
            self.end_headers()

    with HTTPServer(("127.0.0.1", port), Handler) as server:
        print(f"Dashboard: http://127.0.0.1:{port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            if app.process is not None and app.process.poll() is None:
                app.process.terminate()
                try:
                    app.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    app.process.kill()
                    app.process.wait()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    serve(args.runs, args.port)
    return 0
