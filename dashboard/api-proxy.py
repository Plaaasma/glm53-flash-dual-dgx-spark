"""Attribution proxy in front of the vLLM API (on the node clients connect to, usually the one running the socat forwarder).

Clients reach the model on this node's :8888. A NAT redirect (installed by api-proxy.service, removed when it stops, so
traffic falls back to the socat forwarder) sends NEW connections to this proxy instead; it forwards every request
to the head unchanged, streams the response back, and remembers WHO sent each generation request: the client IP,
its Tailscale user and machine (`tailscale whois`), or a LAN name from lan_names.json.

Nothing about a request's content is kept: only who, which path, when it was forwarded, how many engine requests it
creates, its body size, status and end time. The cluster dashboard's collector reads that list from
GET /_proxy/attribution (localhost only) and matches it to the scheduler's requests by arrival time.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import ipaddress
import json
import os
import time
import uuid
from pathlib import Path

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

UPSTREAM = os.environ.get("API_PROXY_UPSTREAM", "http://127.0.0.1:8888")   # the head's API (fabric IP when it is the other node)
LAN_NAMES = Path(os.environ.get("API_PROXY_LAN_NAMES", Path(__file__).with_name("lan_names.json")))
KEEP_S = 1800.0
GEN_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/messages")
HOP = {"connection", "keep-alive", "expect", "proxy-authenticate", "proxy-authorization", "te", "trailers",
       "transfer-encoding", "upgrade", "host", "content-length"}
TAILNET = ipaddress.ip_network("100.64.0.0/10")

client = httpx.AsyncClient(base_url=UPSTREAM, timeout=httpx.Timeout(None, connect=10.0),
                           limits=httpx.Limits(max_connections=None, max_keepalive_connections=64))
records: collections.deque = collections.deque(maxlen=5000)
_who_cache: dict[str, tuple[float, dict]] = {}


def _lan_names() -> dict:
    try:
        return json.loads(LAN_NAMES.read_text())
    except Exception:
        return {}


async def who(ip: str) -> dict:
    hit = _who_cache.get(ip)
    if hit and hit[0] > time.monotonic():
        return hit[1]
    info = {"user": None, "machine": None, "source": "ip"}
    lan = _lan_names().get(ip)
    if lan:
        info = {"user": lan.get("user"), "machine": lan.get("machine"), "source": "lan"}
    else:
        try:
            if ipaddress.ip_address(ip) in TAILNET:
                proc = await asyncio.create_subprocess_exec(
                    "tailscale", "whois", "--json", ip,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
                out, _ = await asyncio.wait_for(proc.communicate(), 5)
                d = json.loads(out)
                prof, node = d.get("UserProfile") or {}, d.get("Node") or {}
                info = {"user": prof.get("DisplayName") or prof.get("LoginName"), "login": prof.get("LoginName"),
                        "machine": (node.get("ComputedName") or node.get("Name") or "").split(".")[0] or None,
                        "source": "tailscale"}
        except Exception:
            pass
    ttl = 600 if info.get("source") != "ip" else 60
    _who_cache[ip] = (time.monotonic() + ttl, info)
    return info


def _engine_requests(body: bytes, path: str) -> int:
    """How many engine requests one HTTP request becomes (multi-prompt completions, n > 1)."""
    try:
        d = json.loads(body)
    except Exception:
        return 1
    n = int(d.get("n") or 1) if isinstance(d, dict) else 1
    if path == "/v1/completions" and isinstance(d, dict):
        p = d.get("prompt")
        if isinstance(p, list) and p and (isinstance(p[0], (str, list))):
            return max(1, len(p)) * n
    return n


async def attribution(request: Request) -> Response:
    if request.client is None or request.client.host not in ("127.0.0.1", "::1"):
        return JSONResponse({"error": "local only"}, status_code=403)
    since = float(request.query_params.get("since", 0) or 0)
    now = time.time()
    out = [r for r in records if (r.get("t_end") or now) >= since]
    return JSONResponse({"now": now, "records": out})


async def proxy(request: Request) -> Response:
    path = request.url.path
    body = await request.body()
    headers = [(k, v) for k, v in request.headers.items() if k.lower() not in HOP]
    rec = None
    if request.method == "POST" and path in GEN_PATHS:
        ip = request.client.host if request.client else "?"
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex
        if not request.headers.get("x-request-id"):
            headers.append(("x-request-id", rid))
        headers.append(("x-forwarded-for", ip))
        rec = {"rid": rid, "ip": ip, "path": path, "t_recv": time.time(), "t_fwd": None, "t_end": None,
               "status": None, "bytes": len(body), "n": _engine_requests(body, path)}
        rec.update({"who": await who(ip)})
        records.append(rec)
    upstream_req = client.build_request(request.method, path, params=request.query_params,
                                        headers=headers, content=body)
    if rec is not None:
        rec["t_fwd"] = time.time()
    # A client that hangs up must abort the upstream request, exactly like closing the socat pipe did (vLLM cancels
    # a request whose connection closes). One watcher covers both phases: waiting for the response head (vLLM
    # sends it only when a non-streaming request finishes) and streaming the body.
    gone = asyncio.Event()

    async def watch():
        while True:
            message = await request.receive()
            if message["type"] == "http.disconnect":
                gone.set()
                return

    watcher = asyncio.create_task(watch())
    sender = asyncio.create_task(client.send(upstream_req, stream=True))
    waiter = asyncio.create_task(gone.wait())
    await asyncio.wait({sender, waiter}, return_when=asyncio.FIRST_COMPLETED)
    if not sender.done():
        sender.cancel()
        watcher.cancel()
        with contextlib.suppress(BaseException):
            await sender
        if rec is not None:
            rec["t_end"], rec["status"] = time.time(), 499
        return Response(status_code=499)
    waiter.cancel()
    try:
        upstream = sender.result()
    except httpx.HTTPError as exc:
        watcher.cancel()
        if rec is not None:
            rec["t_end"], rec["status"] = time.time(), 502
        return JSONResponse({"error": {"message": f"upstream unavailable: {exc}", "type": "proxy_error"}},
                            status_code=502)
    if rec is not None:
        rec["status"] = upstream.status_code
    out_headers = {k: v for k, v in upstream.headers.items() if k.lower() not in HOP - {"content-length"}}

    async def relay():
        chunks = upstream.aiter_raw()
        try:
            while True:
                nxt = asyncio.ensure_future(chunks.__anext__())
                hang = asyncio.ensure_future(gone.wait())
                await asyncio.wait({nxt, hang}, return_when=asyncio.FIRST_COMPLETED)
                hang.cancel()
                if not nxt.done():  # client hung up mid-stream
                    nxt.cancel()
                    with contextlib.suppress(BaseException):
                        await nxt
                    return
                try:
                    yield nxt.result()
                except StopAsyncIteration:
                    return
        finally:
            watcher.cancel()
            await upstream.aclose()
            if rec is not None:
                rec["t_end"] = time.time()

    return StreamingResponse(relay(), status_code=upstream.status_code, headers=out_headers)


async def health(request: Request) -> Response:
    return JSONResponse({"ok": True, "upstream": UPSTREAM, "records": len(records)})


def _prune():
    cutoff = time.time() - KEEP_S
    while records and (records[0].get("t_end") or time.time()) < cutoff:
        records.popleft()


async def _janitor():
    while True:
        await asyncio.sleep(60)
        _prune()


@contextlib.asynccontextmanager
async def _lifespan(app):
    task = asyncio.get_running_loop().create_task(_janitor())
    yield
    task.cancel()
    await client.aclose()


app = Starlette(
    routes=[Route("/_proxy/attribution", attribution), Route("/_proxy/health", health),
            Route("/{path:path}", proxy, methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])],
    lifespan=_lifespan,
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("API_PROXY_HOST", "0.0.0.0"), port=int(os.environ.get("API_PROXY_PORT", "8890")),
                log_level="warning", access_log=False, timeout_keep_alive=30)
