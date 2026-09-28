"""Diagnostics web page (aiohttp, same process and event loop as the Wyoming servers).

  GET  /                 overview: server, memory, endpoints, STT engines, TTS models and their voices, "speak" form
  GET  /bench            benchmark page
  GET  /api/status       everything on the overview as JSON
  POST /api/synthesize   {"text", "voice"} -> audio/wav; timing in X-* headers
  POST /api/bench        {"text", "engines": [...], "max_voices": n, "wer": bool} -> starts a benchmark job
  GET  /api/bench        the job's progress and results
  POST /api/bench/cancel
No authentication: keep the port on a trusted network (like the Wyoming ports). `[server] web_port = 0` turns it off.
"""

import asyncio
import io
import logging
import re
import time
import wave
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
from aiohttp import web
from sentence_stream import SentenceBoundaryDetector

from . import __version__, devices
from .config import Config, EngineConfig
from .engines import create_engine
from .engines.base import SttEngine, SynthesisOptions, TtsPack
from .handler import _prepare_text
from .voices import VoiceRegistry, _PackVoice

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------- helpers


def split_sentences(text: str) -> list[str]:
    sbd = SentenceBoundaryDetector()
    sentences = list(sbd.add_chunk(text))
    rest = sbd.finish()
    if rest.strip():
        sentences.append(rest)
    return [s for s in sentences if s.strip()]


async def speak(engine, text: str, options: SynthesisOptions) -> tuple[bytes, float, float]:
    """Synthesise `text` sentence by sentence like the Wyoming handler; returns (pcm, first audio s, total s).
    For frame-streaming engines "first audio" is the first frame, otherwise the first finished sentence."""
    t = time.perf_counter()
    first: Optional[float] = None
    parts: list[bytes] = []
    stream = getattr(engine, "stream_audio", None)
    for sentence in split_sentences(text):
        prepared = _prepare_text(sentence, engine.config.options.get("auto_punctuation", ".?!"))
        if not prepared:
            continue
        if stream is not None:
            def run(prepared=prepared):
                frames, first_at = [], None
                for pcm in stream(prepared, options):
                    if first_at is None:
                        first_at = time.perf_counter()
                    frames.append(pcm)
                return frames, first_at

            frames, first_at = await asyncio.to_thread(run)
            parts.extend(frames)
            if first is None and first_at is not None:
                first = first_at - t
        else:
            parts.append(await asyncio.to_thread(engine.synthesize, prepared, options))
            if first is None:
                first = time.perf_counter() - t
    total = time.perf_counter() - t
    return b"".join(parts), (first if first is not None else total), total


def wav_bytes(pcm: bytes, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buf.getvalue()


def to16k(pcm: bytes, rate: int) -> np.ndarray:
    x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if rate == 16000 or not len(x):
        return x
    n = int(len(x) * 16000 / rate)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def _words(text: str) -> list[str]:
    return " ".join(re.sub(r"[^\w% ]", " ", text.lower().replace("-", " ")).split()).split()


def word_error_rate(ref: str, hyp: str) -> float:
    """Plain word error rate after lower-casing and dropping punctuation (numbers written as digits vs words count
    as errors, so treat it as a rough check)."""
    r, h = _words(ref), _words(hyp)
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, hw in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
    return d[len(h)] / max(len(r), 1)


# ---------------------------------------------------------------- benchmark job


@dataclass
class BenchRow:
    kind: str  # tts | stt
    engine: str
    model: str
    voice: str = ""
    device: str = ""
    cold_first_s: Optional[float] = None  # load + time to the first sentence's audio (per model)
    warm_first_s: Optional[float] = None
    total_s: Optional[float] = None
    audio_s: Optional[float] = None
    rtf: Optional[float] = None
    wer: Optional[float] = None
    heard: str = ""
    memory_mb: Optional[float] = None
    error: str = ""


@dataclass
class BenchJob:
    text: str
    running: bool = True
    cancelled: bool = False
    started: float = field(default_factory=time.time)
    finished: Optional[float] = None
    done: int = 0
    total: int = 0
    current: str = ""
    rows: list[BenchRow] = field(default_factory=list)


# ---------------------------------------------------------------- the app


class DiagnosticsWeb:
    def __init__(self, config: Config, stt: list[SttEngine], voices: VoiceRegistry) -> None:
        self.config, self.stt, self.voices = config, stt, voices
        self.started = time.time()
        self.job: Optional[BenchJob] = None
        self.task: Optional[asyncio.Task] = None
        self.app = web.Application()
        self.app.add_routes([
            web.get("/", self.page_overview),
            web.get("/bench", self.page_bench),
            web.get("/api/status", self.api_status),
            web.post("/api/synthesize", self.api_synthesize),
            web.post("/api/bench", self.api_bench_start),
            web.get("/api/bench", self.api_bench_state),
            web.post("/api/bench/cancel", self.api_bench_cancel),
        ])

    async def start(self, host: str, port: int) -> web.AppRunner:
        runner = web.AppRunner(self.app, access_log=None)
        await runner.setup()
        await web.TCPSite(runner, host, port).start()
        _LOGGER.info("Diagnostics page on http://%s:%d/", host, port)
        return runner

    # ---- status

    def _engine_row(self, e, kind: str, **extra) -> dict[str, Any]:
        r = e.runtime
        return {"kind": kind, "name": e.name, "backend": e.config.backend, "model": str(e.config.model),
                "device": r.label(), "requested": e.config.device, "detail": r.summary(), "memory_mb": r.memory_mb, **extra}

    def status(self) -> dict[str, Any]:
        v = self.voices
        tts_models = [self._engine_row(e, "fixed voice", loaded=True, voices=[e.name]) for e in v.fixed.values()]
        for lv in v.library_voices.values():
            if lv.engine is not None:
                row = self._engine_row(lv.engine, "folder voice", loaded=True, voices=[lv.name])
            else:
                row = {"kind": "folder voice", "name": lv.name, "backend": lv.library.backend, "model": str(lv.path),
                       "device": "not loaded", "requested": lv.library.device, "detail": "", "memory_mb": None,
                       "loaded": False, "voices": [lv.name]}
            row["folder"] = str(lv.library.path)
            tts_models.append(row)
        for pack in v.packs.values():
            names = [n for n, (p, _vid) in v.pack_voices.items() if p is pack]
            row = self._engine_row(pack.engine, "voice pack", loaded=pack.loaded, voices=names)
            if not pack.loaded:
                row["device"] = "not loaded" if not pack.failed else f"failed: {pack.failed}"
            tts_models.append(row)
        endpoints = [{"uri": ep.uri, "stt": ep.stt if ep.stt is not None else [e.name for e in self.stt], "tts": ep.tts}
                     for ep in self.config.endpoints]
        return {
            "version": __version__, "uptime_s": round(time.time() - self.started),
            "memory": {"process_rss_mb": round(devices.rss_mb()), "container_mb": devices.cgroup_memory_mb()},
            "endpoints": endpoints,
            "stt": [self._engine_row(e, "stt", languages=e.languages) for e in self.stt],
            "tts_models": tts_models,
            "voices": [asdict(x) for x in v.voices()],
            "bench": {"running": bool(self.job and self.job.running)},
        }

    async def api_status(self, request: web.Request) -> web.Response:
        return web.json_response(self.status())

    # ---- sample synthesis

    async def api_synthesize(self, request: web.Request) -> web.Response:
        body = await request.json()
        text = str(body.get("text") or "").strip()
        if not text:
            raise web.HTTPBadRequest(text="text is empty")
        async with self.voices.use(body.get("voice") or None) as engine:
            pcm, first, total = await speak(engine, text, SynthesisOptions())
            rate = engine.sample_rate
            label = engine.runtime.label()
        seconds = len(pcm) / 2 / rate
        return web.Response(body=wav_bytes(pcm, rate), content_type="audio/wav", headers={
            "X-First-Audio": f"{first:.3f}", "X-Total": f"{total:.3f}", "X-Audio-Seconds": f"{seconds:.2f}",
            "X-RTF": f"{total / max(seconds, 1e-6):.2f}", "X-Device": label, "X-Voice": getattr(engine, "name", ""),
            "Access-Control-Expose-Headers": "X-First-Audio, X-Total, X-Audio-Seconds, X-RTF, X-Device, X-Voice",
        })

    # ---- benchmark

    async def api_bench_start(self, request: web.Request) -> web.Response:
        if self.job and self.job.running:
            raise web.HTTPConflict(text="a benchmark is already running")
        body = await request.json()
        text = str(body.get("text") or "").strip()
        if not text:
            raise web.HTTPBadRequest(text="text is empty")
        engines = set(body.get("engines") or ["piper", "folders", "kokoro", "kitten", "pocket", "stt"])
        max_voices = int(body.get("max_voices") or 0)
        self.job = BenchJob(text=text)
        self.task = asyncio.create_task(self._bench(self.job, engines, max_voices, bool(body.get("wer", True))))
        return web.json_response({"started": True})

    async def api_bench_state(self, request: web.Request) -> web.Response:
        job = self.job
        if job is None:
            return web.json_response({"job": None})
        return web.json_response({"job": {**{k: v for k, v in asdict(job).items() if k != "rows"},
                                          "rows": [asdict(r) for r in job.rows]}})

    async def api_bench_cancel(self, request: web.Request) -> web.Response:
        if self.job and self.job.running:
            self.job.cancelled = True
        return web.json_response({"cancelled": True})

    def _plan(self, engines: set[str], max_voices: int) -> list[tuple[str, EngineConfig, list[tuple[str, Optional[str]]]]]:
        """(kind, engine config, [(voice name, pack voice id or None)]) per model to benchmark."""
        v = self.voices
        plan = []
        if "piper" in engines:
            for e in v.fixed.values():
                plan.append(("tts", e.config, [(e.name, None)]))
        if "folders" in engines:
            for lv in v.library_voices.values():
                cfg = EngineConfig(kind="tts", name=lv.name, backend=lv.library.backend, model=lv.path, device=lv.library.device,
                                   languages=lv.languages, options=v.speech_options(lv.name, lv.library.options))
                plan.append(("tts", cfg, [(lv.name, None)]))
        for pack in v.packs.values():
            if pack.engine.config.backend in engines or pack.engine.name in engines:
                names = [(n, vid) for n, (p, vid) in v.pack_voices.items() if p is pack]
                plan.append(("tts", pack.engine.config, names[:max_voices] if max_voices else names))
        if "stt" in engines:
            for e in self.stt:
                plan.append(("stt", e.config, [("", None)]))
        return plan

    async def _bench(self, job: BenchJob, engines: set[str], max_voices: int, with_wer: bool) -> None:
        try:
            plan = self._plan(engines, max_voices)
            job.total = sum(len(voices) for _, _, voices in plan)
            stt_audio: Optional[np.ndarray] = None
            for kind, cfg, voices in plan:
                if job.cancelled:
                    break
                if kind == "stt":
                    if stt_audio is None:  # the benchmark text spoken by the default voice
                        job.current = "synthesising the text for the STT runs"
                        async with self.voices.use(None) as e:
                            pcm, _, _ = await speak(e, job.text, SynthesisOptions())
                            stt_audio = to16k(pcm, e.sample_rate)
                    await self._bench_stt(job, cfg, stt_audio)
                else:
                    await self._bench_tts(job, cfg, voices, with_wer)
        except Exception as err:  # keep the page alive
            _LOGGER.exception("benchmark failed")
            job.rows.append(BenchRow(kind="-", engine="-", model="-", error=str(err)))
        finally:
            job.running, job.finished, job.current = False, time.time(), ""

    async def _fresh(self, cfg: EngineConfig):
        """A separate engine instance (not the one serving requests), so "cold" really includes loading."""
        engine = create_engine(cfg, self.config.gpu)
        rss0 = devices.memory_mb()
        t = time.perf_counter()
        await asyncio.to_thread(engine.load)
        return engine, time.perf_counter() - t, rss0

    async def _bench_tts(self, job: BenchJob, cfg: EngineConfig, voices, with_wer: bool) -> None:
        job.current = f"loading {cfg.backend} {cfg.name}"
        try:
            engine, load_s, rss0 = await self._fresh(cfg)
        except Exception as err:
            for name, _ in voices:
                job.rows.append(BenchRow(kind="tts", engine=cfg.backend, model=cfg.model.name, voice=name, error=f"load: {err}"))
                job.done += 1
            return
        try:
            first_voice = True
            for name, vid in voices:
                if job.cancelled:
                    break
                job.current = f"{cfg.backend} {name}"
                row = BenchRow(kind="tts", engine=cfg.backend, model=cfg.model.name, voice=name)
                try:
                    target = (_PackVoice(engine, vid, name, self.voices.speech_options(name, cfg.options))
                              if isinstance(engine, TtsPack) else engine)
                    if first_voice:  # cold: load + the first sentence of the first voice (includes shader compiles)
                        first_sentence = (split_sentences(job.text) or [job.text])[0]
                        _, first, _ = await speak(target, first_sentence, SynthesisOptions())
                        row.cold_first_s = round(load_s + first, 3)
                        row.memory_mb = round(devices.memory_mb() - rss0)
                        first_voice = False
                    pcm, first, total = await speak(target, job.text, SynthesisOptions())
                    row.device = engine.runtime.label()
                    row.warm_first_s, row.total_s = round(first, 3), round(total, 3)
                    row.audio_s = round(len(pcm) / 2 / target.sample_rate, 2)
                    row.rtf = round(total / max(row.audio_s, 1e-6), 2)
                    if with_wer and self.stt:
                        heard = await asyncio.to_thread(self.stt[0].transcribe, to16k(pcm, target.sample_rate), "en")
                        row.heard, row.wer = heard, round(word_error_rate(job.text, heard), 3)
                except Exception as err:
                    row.error = str(err) or type(err).__name__
                job.rows.append(row)
                job.done += 1
        finally:
            await asyncio.to_thread(engine.close)

    async def _bench_stt(self, job: BenchJob, cfg: EngineConfig, audio: np.ndarray) -> None:
        job.current = f"loading {cfg.backend} {cfg.name}"
        row = BenchRow(kind="stt", engine=cfg.backend, model=cfg.model.name, voice=cfg.name)
        seconds = len(audio) / 16000
        try:
            engine, load_s, rss0 = await self._fresh(cfg)
            try:
                t = time.perf_counter()
                await asyncio.to_thread(engine.transcribe, audio, None)
                row.cold_first_s = round(load_s + time.perf_counter() - t, 3)
                row.memory_mb = round(devices.memory_mb() - rss0)
                t = time.perf_counter()
                heard = await asyncio.to_thread(engine.transcribe, audio, None)
                took = time.perf_counter() - t
                row.device = engine.runtime.label()
                row.warm_first_s = row.total_s = round(took, 3)
                row.audio_s, row.rtf = round(seconds, 2), round(took / max(seconds, 1e-6), 2)
                row.heard, row.wer = heard, round(word_error_rate(job.text, heard), 3)
            finally:
                await asyncio.to_thread(engine.close)
        except Exception as err:
            row.error = str(err) or type(err).__name__
        job.rows.append(row)
        job.done += 1

    # ---- pages

    async def page_overview(self, request: web.Request) -> web.Response:
        return web.Response(text=_PAGE.replace("%BODY%", _OVERVIEW), content_type="text/html")

    async def page_bench(self, request: web.Request) -> web.Response:
        return web.Response(text=_PAGE.replace("%BODY%", _BENCH), content_type="text/html")


_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>wyoming-vulkan diagnostics</title><style>
:root{--bg:#fff;--fg:#1d2330;--mute:#667085;--line:#e3e6ec;--card:#f7f8fa;--acc:#2f6fed;--bad:#c0362c;--ok:#1f8a4c}
@media (prefers-color-scheme:dark){:root{--bg:#14171c;--fg:#e6e9ef;--mute:#9aa3b2;--line:#2a2f38;--card:#1b1f26;--acc:#6d9bff;--bad:#ff7b72;--ok:#56d364}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1200px;margin:0 auto;padding:16px}nav a{margin-right:16px;color:var(--acc);text-decoration:none;font-weight:600}
h1{font-size:20px;margin:8px 0 4px}h2{font-size:16px;margin:24px 0 8px}.mute{color:var(--mute)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}
th{cursor:pointer;white-space:nowrap}td.n{text-align:right;font-variant-numeric:tabular-nums}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:8px 0}
textarea,select,input{font:inherit;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px}
textarea{width:100%;box-sizing:border-box}button{font:inherit;background:var(--acc);color:#fff;border:0;border-radius:6px;padding:7px 14px;cursor:pointer}
button[disabled]{opacity:.5}.bad{color:var(--bad)}.ok{color:var(--ok)}details summary{cursor:pointer;color:var(--acc)}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden}.bar>div{height:100%;background:var(--acc);width:0}
.wrap{overflow-x:auto}</style></head><body><main>
<nav><a href="/">Overview</a><a href="/bench">Benchmark</a></nav>%BODY%</main></body></html>"""

_OVERVIEW = """
<h1>wyoming-vulkan <span id="ver" class="mute"></span></h1><div id="server" class="mute"></div>
<h2>Speak</h2><div class="card"><textarea id="text" rows="3">The front door is locked and the lights in the living room are off.</textarea>
<p><select id="voice"></select> <button id="go">Speak</button> <span id="timing" class="mute"></span></p><audio id="player" controls></audio></div>
<h2>Endpoints</h2><div class="wrap"><table id="ep"></table></div>
<h2>Speech-to-text engines</h2><div class="wrap"><table id="stt"></table></div>
<h2>Text-to-speech models</h2><p class="mute">Memory = growth of the container's memory while the model loaded and warmed up (approximate: includes the iGPU's buffers, which live in system RAM, and the page cache of the model file; loads that overlap blur it).</p><div class="wrap"><table id="tts"></table></div>
<script>
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const mb=v=>v==null?'':Math.round(v)+' MiB';
async function load(){const s=await (await fetch('/api/status')).json();
 document.getElementById('ver').textContent=s.version;
 document.getElementById('server').textContent=`up ${Math.round(s.uptime_s/60)} min · process ${s.memory.process_rss_mb} MiB · container ${s.memory.container_mb?Math.round(s.memory.container_mb)+' MiB':'n/a'} · ${s.voices.length} voices`;
 document.getElementById('ep').innerHTML='<tr><th>Wyoming endpoint</th><th>STT engines</th><th>Voices</th></tr>'+s.endpoints.map(e=>`<tr><td>${esc(e.uri)}</td><td>${esc(e.stt.join(', ')||'none')}</td><td>${e.tts?'all':'none'}</td></tr>`).join('');
 document.getElementById('stt').innerHTML='<tr><th>Name</th><th>Backend</th><th>Model</th><th>Runs on</th><th>Memory</th><th>Languages</th></tr>'+s.stt.map(e=>`<tr><td>${esc(e.name)}</td><td>${esc(e.backend)}</td><td>${esc(e.model.split('/').pop())}</td><td>${esc(e.device)}</td><td class="n">${mb(e.memory_mb)}</td><td class="mute">${esc(e.languages.length>8?e.languages.length+' languages':e.languages.join(', '))}</td></tr>`).join('');
 document.getElementById('tts').innerHTML='<tr><th>Model</th><th>Kind</th><th>Backend</th><th>Runs on</th><th>Loaded</th><th>Memory</th><th>Voices</th></tr>'+s.tts_models.map(m=>`<tr><td>${esc(m.model.split('/').pop())}</td><td>${esc(m.kind)}</td><td>${esc(m.backend)}</td><td>${esc(m.device)}</td><td>${m.loaded?'<span class="ok">yes</span>':'no'}</td><td class="n">${mb(m.memory_mb)}</td><td>${m.voices.length>1?`<details><summary>${m.voices.length} voices</summary>${m.voices.map(esc).join(', ')}</details>`:esc(m.voices[0])}</td></tr>`).join('');
 const sel=document.getElementById('voice'),cur=sel.value,groups={};
 s.voices.forEach(v=>{const g=v.description.split(' ')[0];(groups[g]=groups[g]||[]).push(v)});
 sel.innerHTML=Object.entries(groups).map(([g,vs])=>`<optgroup label="${esc(g)}">`+vs.map(v=>`<option value="${esc(v.name)}">${esc(v.description)} · ${esc(v.languages.join(','))}</option>`).join('')+'</optgroup>').join('');
 if(cur)sel.value=cur;}
document.getElementById('go').onclick=async()=>{const b=document.getElementById('go');b.disabled=true;const t=document.getElementById('timing');t.textContent='synthesising…';
 try{const r=await fetch('/api/synthesize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:document.getElementById('text').value,voice:document.getElementById('voice').value})});
 if(!r.ok)throw new Error(await r.text());const h=k=>r.headers.get(k);document.getElementById('player').src=URL.createObjectURL(await r.blob());document.getElementById('player').play();
 t.textContent=`${h('X-Voice')} on ${h('X-Device')}: first audio ${h('X-First-Audio')} s, total ${h('X-Total')} s for ${h('X-Audio-Seconds')} s audio (RTF ${h('X-RTF')})`;load();}
 catch(e){t.innerHTML='<span class="bad">'+esc(e.message)+'</span>'}finally{b.disabled=false}};
load();setInterval(load,15000);
</script>"""

_BENCH = """
<h1>Benchmark</h1><p class="mute">Runs the text through every selected model and voice with a separate, freshly loaded copy of each model
(so <b>cold</b> = load + first sentence), then again warm. <b>First</b> = time to the first sentence's audio (first frame for streaming engines).
<b>RTF</b> = synthesis time / audio length (below 1 = faster than real time). <b>WER</b> = the audio transcribed by the first STT engine
(rough: digits vs words count as errors). STT rows transcribe the text spoken by the default voice. The benchmark shares the GPU with live requests.</p>
<div class="card"><textarea id="text" rows="3">Good evening. The front door is locked, the lights downstairs are off, and tomorrow will be mostly cloudy with a high of eighteen degrees.</textarea>
<p id="engines"></p><p>Voices per voice pack: <input id="max" type="number" min="0" value="3" style="width:5em"> (0 = all) &nbsp;
<label><input type="checkbox" id="wer" checked> word error rate</label> &nbsp; <button id="start">Start</button> <button id="cancel">Cancel</button></p>
<div class="bar"><div id="bar"></div></div><p id="state" class="mute"></p></div>
<p><button id="csv">Download CSV</button></p><div class="wrap"><table id="res"></table></div>
<script>
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const ENG=[['piper','Piper (fixed voices)'],['folders','Piper folder voices'],['kokoro','Kokoro'],['kitten','Kitten'],['pocket','Pocket'],['stt','Speech-to-text']];
document.getElementById('engines').innerHTML=ENG.map(([k,l])=>`<label><input type="checkbox" value="${k}" ${k!=='folders'?'checked':''}> ${l}</label>`).join(' &nbsp; ');
const COLS=[['kind','Kind'],['engine','Engine'],['voice','Voice / model'],['device','Runs on'],['cold_first_s','Cold first (s)'],['warm_first_s','Warm first (s)'],['total_s','Total (s)'],['audio_s','Audio (s)'],['rtf','RTF'],['wer','WER'],['memory_mb','Memory (MiB)'],['heard','Heard'],['error','Error']];
let rows=[],sortKey=null,sortDir=1;
function render(){const r=[...rows];if(sortKey)r.sort((a,b)=>((a[sortKey]??1e9)>(b[sortKey]??1e9)?1:-1)*sortDir);
 document.getElementById('res').innerHTML='<tr>'+COLS.map(([k,l])=>`<th data-k="${k}">${l}</th>`).join('')+'</tr>'+r.map(x=>'<tr>'+COLS.map(([k])=>{let v=x[k];if(k==='wer'&&v!=null)v=(v*100).toFixed(1)+' %';if(k==='voice'&&!v)v=x.model;
  return `<td class="${typeof x[k]==='number'?'n':''} ${k==='error'&&v?'bad':''} ${k==='heard'?'mute':''}">${esc(v??'')}</td>`}).join('')+'</tr>').join('');
 document.querySelectorAll('#res th').forEach(th=>th.onclick=()=>{sortDir=sortKey===th.dataset.k?-sortDir:1;sortKey=th.dataset.k;render()});}
async function poll(){const s=await (await fetch('/api/bench')).json();const j=s.job;if(!j){document.getElementById('state').textContent='no benchmark yet';return}
 rows=j.rows;render();document.getElementById('bar').style.width=(j.total?100*j.done/j.total:0)+'%';
 document.getElementById('state').textContent=j.running?`running ${j.done}/${j.total}: ${j.current}`:`finished ${j.done}/${j.total}${j.cancelled?' (cancelled)':''} in ${Math.round((j.finished-j.started))} s`;
 document.getElementById('start').disabled=j.running;if(j.running)setTimeout(poll,1000);}
document.getElementById('start').onclick=async()=>{const engines=[...document.querySelectorAll('#engines input:checked')].map(i=>i.value);
 const r=await fetch('/api/bench',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:document.getElementById('text').value,engines,max_voices:+document.getElementById('max').value,wer:document.getElementById('wer').checked})});
 if(!r.ok){document.getElementById('state').textContent=await r.text();return}poll();};
document.getElementById('cancel').onclick=()=>fetch('/api/bench/cancel',{method:'POST'});
document.getElementById('csv').onclick=()=>{const k=COLS.map(c=>c[0]);const csv=[k.join(',')].concat(rows.map(x=>k.map(c=>JSON.stringify(x[c]??'')).join(','))).join('\\n');
 const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv'}));a.download='wyoming-vulkan-benchmark.csv';a.click();};
poll();
</script>"""
