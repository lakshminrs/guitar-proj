"use client";
import { useParams, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import { api, type Chord, type Exercise } from "@/lib/api";
import { makeScript, type ScriptItem } from "@/lib/collect";
import { COLORS, drawGrid, drawHand, fillCell, fingertipCells, label, mapper, ring, splitHands } from "@/lib/draw";
import { CALIB_POINTS, FINGER_NAMES, homographyFromCalib, type CalibName, type Mat3, type Pt } from "@/lib/geometry";
import { getHandLandmarker } from "@/lib/hands";
import { Metronome } from "@/lib/metronome";

type Phase = "idle" | "starting" | "live" | "countin" | "recording" | "confirm" | "uploading";
type Take = { blob: Blob; clicks_ms: number[]; latency: number; mime: string };
type Tags = { player: string; guitar: string; mic: string; room: string; audio: string };
const TAGS_KEY = "gc.collect.tags.v1";
const COUNT_IN = 4;
const TAP_HELP: Record<CalibName, string> = {
  nut_low: "Tap the NUT on the thick (low E) string side",
  nut_high: "Tap the NUT on the thin (high E) string side",
  f12_low: "Tap the 12th FRET (double dot) on the low E side",
  f12_high: "Tap the 12th FRET on the high E side",
};
const CAL_KEY = "gc.calibration.v1";

function pickMime() {
  for (const m of ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm", "video/mp4"]) {
    if (typeof MediaRecorder !== "undefined" && MediaRecorder.isTypeSupported(m)) return m;
  }
  return "";
}

export default function Practice() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const [ex, setEx] = useState<Exercise | null>(null);
  const [chords, setChords] = useState<Record<string, Chord>>({});
  const [phase, setPhase] = useState<Phase>("idle");
  const [msg, setMsg] = useState<string>("");
  const [mirror, setMirror] = useState(true);
  const [calib, setCalib] = useState<Partial<Record<CalibName, Pt>>>({});
  const [tapping, setTapping] = useState(false);
  const [preview, setPreview] = useState<string>("");
  const [hud, setHud] = useState<{ big: string; small: string; cue?: string } | null>(null);
  // scripted-take (data collection) mode: /practice/<id>?collect=1
  const [collect, setCollect] = useState(false);
  const [script, setScript] = useState<ScriptItem[]>([]);
  const [tags, setTags] = useState<Tags>({ player: "", guitar: "acoustic", mic: "laptop", room: "", audio: "speakers" });
  const [usable, setUsable] = useState(true);
  const [take, setTake] = useState<Take | null>(null);
  const [tips, setTips] = useState<ReturnType<typeof fingertipCells>>([]);
  const [fps, setFps] = useState(0);

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const ctxRef = useRef<AudioContext | null>(null);
  const recRef = useRef<{ rec: MediaRecorder; met: Metronome } | null>(null);
  const live = useRef({ H: null as Mat3 | null, mirror: true, chord: "" as string, calib: {} as Partial<Record<CalibName, Pt>> });

  const H = homographyFromCalib(calib);
  live.current.H = H;
  live.current.mirror = mirror;
  live.current.calib = calib;

  useEffect(() => {
    api.exercise(Number(id)).then((e) => { setEx(e); setPreview(e.bars[0].chord); }).catch((e) => setMsg(String(e)));
    api.chords().then((cs) => setChords(Object.fromEntries(cs.map((c) => [c.name, c])))).catch(() => {});
    setCollect(new URLSearchParams(window.location.search).get("collect") === "1");
    try {
      const t = localStorage.getItem(TAGS_KEY);
      if (t) setTags((o) => ({ ...o, ...JSON.parse(t) }));
    } catch { /* ignore */ }
    try {
      const saved = localStorage.getItem(CAL_KEY);
      if (saved) setCalib(JSON.parse(saved));
    } catch { /* storage unavailable */ }
  }, [id]);

  useEffect(() => { if (phase === "live") live.current.chord = preview; }, [preview, phase]);
  useEffect(() => () => { streamRef.current?.getTracks().forEach((t) => t.stop()); ctxRef.current?.close(); }, []);

  // ------------------------------------------------------------------ camera + render loop
  const start = useCallback(async () => {
    setPhase("starting");
    setMsg("Loading hand tracker…");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 } },
        // raw audio: no echo cancellation / noise suppression / AGC (they wreck music)
        audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      });
      streamRef.current = stream;
      ctxRef.current = new AudioContext({ latencyHint: "interactive" });
      const v = videoRef.current!;
      v.srcObject = stream;
      await v.play();
      const lmk = await getHandLandmarker();
      setMsg("");
      setPhase("live");

      const mobile = /Android|iPhone|iPad/i.test(navigator.userAgent);
      const minDt = 1000 / (mobile ? 15 : 30);
      let last = 0, frames = 0, fpsT = performance.now(), tipsT = 0;
      const loop = () => {
        if (!streamRef.current) return;
        requestAnimationFrame(loop);
        const now = performance.now();
        if (now - last < minDt || v.readyState < 2) return;
        last = now;
        const c = canvasRef.current!;
        if (c.width !== v.videoWidth) { c.width = v.videoWidth; c.height = v.videoHeight; }
        const g = c.getContext("2d")!;
        g.clearRect(0, 0, c.width, c.height);
        const m = mapper(c.width, c.height, live.current.mirror);
        const { H, chord, calib } = live.current;

        // calibration taps
        for (const k of CALIB_POINTS) {
          const p = calib[k];
          if (p) { const q = m(p); g.fillStyle = COLORS.nut; g.beginPath(); g.arc(q[0], q[1], 6, 0, 7); g.fill(); }
        }
        if (H) {
          drawGrid(g, H, m);
          const shape = chord ? chordsRef.current[chord] : null;
          if (shape) {
            shape.frets.forEach((f, i) => {
              if (f > 0) fillCell(g, H, m, 6 - i, f, COLORS.target, COLORS.targetStroke);
            });
          }
        }
        const res = lmk.detectForVideo(v, now);
        const hands: Pt[][] = res.landmarks.map((h) => h.map((p) => [p.x, p.y] as Pt));
        const { fret, strum } = splitHands(hands, H);
        if (strum) drawHand(g, strum, m, COLORS.strumHand);
        if (fret) {
          drawHand(g, fret, m, COLORS.fretHand);
          if (H) {
            const cells = fingertipCells(fret, H);
            const shape = chord ? chordsRef.current[chord] : null;
            for (const t of cells) {
              if (!t.onNeck) continue;
              const target = shape ? shape.frets.findIndex((f, i) => f > 0 && shape.fingers[i] === t.finger) : -1;
              const ok = target >= 0 && 6 - target === t.string && shape!.frets[target] === t.fret;
              const p = m(t.tip);
              ring(g, p, target < 0 ? "#fff" : ok ? COLORS.targetStroke : COLORS.error);
              label(g, p, `${t.finger}: s${t.string} f${t.fret}`, ok ? "rgba(20,90,50,0.85)" : "rgba(0,0,0,0.75)");
            }
            if (now - tipsT > 200) { tipsT = now; setTips(cells); }
          }
        } else if (now - tipsT > 200) { tipsT = now; setTips([]); }
        frames++;
        if (now - fpsT > 1000) { setFps(frames); frames = 0; fpsT = now; }
      };
      requestAnimationFrame(loop);
    } catch (e) {
      setPhase("idle");
      setMsg(`Couldn't start the camera/mic: ${e}`);
    }
  }, []);

  const chordsRef = useRef(chords);
  chordsRef.current = chords;

  // ------------------------------------------------------------------ calibration taps
  const onTap = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!tapping) return;
    const c = canvasRef.current!, r = c.getBoundingClientRect();
    const s = Math.min(r.width / c.width, r.height / c.height);
    const dw = c.width * s, dh = c.height * s;
    let x = (e.clientX - r.left - (r.width - dw) / 2) / dw;
    const y = (e.clientY - r.top - (r.height - dh) / 2) / dh;
    if (x < 0 || x > 1 || y < 0 || y > 1) return;
    if (mirror) x = 1 - x; // store raw (as-recorded) coordinates
    const next = CALIB_POINTS.find((k) => !calib[k]);
    if (!next) return;
    const nc = { ...calib, [next]: [x, y] as Pt };
    setCalib(nc);
    if (CALIB_POINTS.every((k) => nc[k])) {
      setTapping(false);
      try { localStorage.setItem(CAL_KEY, JSON.stringify(nc)); } catch { /* ignore */ }
    }
  };
  const nextTap = CALIB_POINTS.find((k) => !calib[k]);

  // ------------------------------------------------------------------ record a take
  const record = async () => {
    if (!ex || !streamRef.current || !ctxRef.current) return;
    const ctx = ctxRef.current;
    await ctx.resume();
    const bpb = parseInt(ex.time_sig.split("/")[0], 10);
    const nBeats = ex.bars.length * bpb;
    const spb = 60 / ex.bpm;
    const mime = pickMime();
    const rec = new MediaRecorder(streamRef.current, mime ? { mimeType: mime, videoBitsPerSecond: 4_000_000 } : undefined);
    const chunks: Blob[] = [];
    let recStart = 0;
    const met = new Metronome(ctx);
    rec.ondataavailable = (ev) => { if (ev.data.size) chunks.push(ev.data); };
    rec.onstart = () => {
      recStart = ctx.currentTime;
      setPhase("countin");
      met.start(ex.bpm, COUNT_IN + nBeats, bpb, ctx.currentTime + 0.5, (b) => {
        if (b < COUNT_IN) { setHud({ big: String(COUNT_IN - b), small: `count-in · first chord ${ex.bars[0].chord}` }); return; }
        setPhase("recording");
        const k = b - COUNT_IN, bar = Math.floor(k / bpb);
        const chord = ex.bars[bar]?.chord ?? "";
        live.current.chord = chord;
        const next = ex.bars[bar + 1]?.chord;
        // scripted take: warn a bar ahead, then show the instruction during its bar
        const now = collect ? script.filter((s) => s.bar === bar + 1) : [];
        const soon = collect ? script.filter((s) => s.bar === bar + 2) : [];
        const cue = now.length ? `NOW · beat ${now[0].beatLabel}: ${now[0].text}`
          : soon.length ? `NEXT BAR · beat ${soon[0].beatLabel}: ${soon[0].text}` : undefined;
        setHud({ big: chord, small: `bar ${bar + 1}/${ex.bars.length} · beat ${(k % bpb) + 1}${next ? ` · next ${next}` : ""} · ${ex.bars[bar]?.strum_pattern}`, cue });
      }, () => window.setTimeout(() => rec.stop(), spb * 1000 + 300));
    };
    rec.onstop = async () => {
      setHud(null);
      const blob = new Blob(chunks, { type: (rec.mimeType || "video/webm").split(";")[0] });
      const latency = ((ctx.outputLatency || 0) + (ctx.baseLatency || 0)) * 1000;
      const t: Take = { blob, latency, mime: rec.mimeType, clicks_ms: met.clickTimes.map((c) => (c - recStart) * 1000 + latency) };
      if (collect) {            // let the player say which scripted mistakes they actually made
        setTake(t);
        setUsable(true);
        setPhase("confirm");
        return;
      }
      submit(t);
    };
    recRef.current = { rec, met };
    rec.start(1000);
  };

  const submit = async (t: Take) => {
    if (!ex) return;
    setPhase("uploading");
    try {
      setMsg("Uploading take…");
      const s = await api.createSession(ex.id);
      await api.upload(s.upload_url, t.blob);
      setMsg("Queued for analysis…");
      const cal = homographyFromCalib(calib) ? { points: calib, mirrored_preview: mirror } : null;
      await api.analyze(s.id, {
        clicks_ms: t.clicks_ms, count_in_beats: COUNT_IN, calibration: cal,
        device: { ua: navigator.userAgent, mime: t.mime, output_latency_ms: t.latency,
                  video: streamRef.current?.getVideoTracks()[0]?.getSettings() },
        collect: collect ? {
          labels: script.filter((s) => s.done).map((s) => s.label),
          script: script.map((s) => ({ bar: s.bar, beat: s.beatLabel, text: s.text, done: s.done, label: s.label })),
          usable, tags,
        } : undefined,
      });
      if (collect) {            // stay here and deal the next script: takes go much faster
        setTake(null);
        setScript(makeScript(ex, chords));
        setMsg(`Saved take ${s.id.slice(0, 6)}${usable ? "" : " (marked unusable)"}. New script below.`);
        setPhase("live");
        return;
      }
      router.push(`/sessions/${s.id}`);
    } catch (e) {
      setMsg(`Upload failed: ${e}`);
      setPhase(collect && take ? "confirm" : "live");
    }
  };

  useEffect(() => {
    if (collect && ex && Object.keys(chords).length && script.length === 0 && phase !== "confirm") setScript(makeScript(ex, chords));
    // deal automatically only on first load; afterwards "Deal a different script" and each submit deal
  }, [collect, ex, chords]);

  const saveTags = (t: Tags) => {
    setTags(t);
    try { localStorage.setItem(TAGS_KEY, JSON.stringify(t)); } catch { /* ignore */ }
  };

  const stopEarly = () => {
    const r = recRef.current;
    if (!r || r.rec.state === "inactive") return;
    r.met.stop();
    r.rec.stop();
    setMsg("Stopped early — grading what you played…");
  };

  const shape = chords[preview];
  return (
    <main className="wrap">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
        <div>
          <h1>{ex?.title ?? "…"}</h1>
          <div className="muted small">{ex ? `${ex.bpm} BPM · ${ex.time_sig} · ${ex.bars.map((b) => b.chord).join(" – ")}` : ""}</div>
        </div>
        <div className="steps">
          <span className={phase !== "idle" ? "on" : ""}>1 · camera</span>
          <span className={H ? "on" : ""}>2 · calibrate</span>
          <span className={phase === "recording" || phase === "countin" ? "on" : ""}>3 · play</span>
          <span className={phase === "uploading" ? "on" : ""}>4 · analyze</span>
        </div>
      </div>

      <div className="layout">
        <div>
          <div className="stage">
            <video ref={videoRef} className={mirror ? "mirror" : ""} playsInline muted />
            <canvas ref={canvasRef} className={tapping ? "tap" : ""} onClick={onTap} />
            {hud && <div className="hud"><div className="big">{hud.big}</div><div>{hud.small}</div>
              {hud.cue && <div style={{ marginTop: 6, color: hud.cue.startsWith("NOW") ? "var(--bad)" : "var(--accent)", fontWeight: 700, maxWidth: 420 }}>{hud.cue}</div>}
            </div>}
            {tapping && nextTap && <div className="hud" style={{ top: "auto", bottom: 12 }}>{TAP_HELP[nextTap]}</div>}
            {phase === "idle" && (
              <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center" }}>
                <button className="primary" onClick={start}>Start camera + mic</button>
              </div>
            )}
          </div>
          {msg && <p className="muted">{msg}</p>}
          <div className="row" style={{ marginTop: 12 }}>
            <button onClick={() => { setCalib({}); setTapping(true); }} disabled={phase !== "live"}>
              {H ? "Re-calibrate fretboard" : "Calibrate fretboard (4 taps)"}
            </button>
            <button onClick={() => setMirror((m) => !m)}>{mirror ? "Mirrored preview" : "Unmirrored preview"}</button>
            <button className="primary" onClick={record} disabled={phase !== "live"}>
              ● Record take {H ? "" : "(audio + hands only)"}
            </button>
            {(phase === "recording" || phase === "countin") && <button className="danger" onClick={stopEarly}>■ Stop</button>}
            <span className="muted small">{phase !== "idle" ? `${fps} fps` : ""}</span>
          </div>
        </div>

        <aside className="grid">
          {collect && (
            <div className="panel" style={{ borderColor: "var(--accent)" }}>
              <h2>Scripted take <span className="muted small">(recording data)</span></h2>
              <div className="grid small" style={{ gridTemplateColumns: "auto 1fr", gap: "6px 8px", alignItems: "center" }}>
                <label htmlFor="t-player">Player</label>
                <input id="t-player" value={tags.player} placeholder="name or initials" disabled={phase !== "live" && phase !== "idle"}
                       onChange={(e) => saveTags({ ...tags, player: e.target.value })} />
                <label htmlFor="t-guitar">Guitar</label>
                <select id="t-guitar" value={tags.guitar} onChange={(e) => saveTags({ ...tags, guitar: e.target.value })}>
                  <option value="acoustic">acoustic</option><option value="electro-acoustic">electro-acoustic</option>
                  <option value="electric">electric (amp)</option><option value="classical">classical / nylon</option>
                </select>
                <label htmlFor="t-mic">Device</label>
                <select id="t-mic" value={tags.mic} onChange={(e) => saveTags({ ...tags, mic: e.target.value })}>
                  <option value="laptop">laptop</option><option value="phone">phone</option><option value="tablet">tablet</option>
                  <option value="webcam">external webcam / mic</option>
                </select>
                <label htmlFor="t-audio">Click via</label>
                <select id="t-audio" value={tags.audio} onChange={(e) => saveTags({ ...tags, audio: e.target.value })}>
                  <option value="speakers">speakers</option><option value="headphones">headphones</option>
                </select>
                <label htmlFor="t-room">Room</label>
                <input id="t-room" value={tags.room} placeholder="e.g. bedroom, kitchen"
                       onChange={(e) => saveTags({ ...tags, room: e.target.value })} />
              </div>

              {phase !== "confirm" && (
                <>
                  <h3 style={{ marginTop: 12 }}>Mistakes to make this take</h3>
                  {script.length === 0
                    ? <p className="small">None: <b>play it as cleanly as you can.</b></p>
                    : <ol className="small" style={{ paddingLeft: 18, margin: 0 }}>
                        {script.map((s) => <li key={s.id} style={{ marginBottom: 6 }}><b>Bar {s.bar}, beat {s.beatLabel}:</b> {s.text}</li>)}
                      </ol>}
                  <p className="muted small">Only these. Play everything else as well as you can. The cue also appears on the video a bar early.</p>
                  <button onClick={() => ex && setScript(makeScript(ex, chords))} disabled={phase !== "live" && phase !== "idle"}>Deal a different script</button>
                </>
              )}

              {phase === "confirm" && take && (
                <>
                  <h3 style={{ marginTop: 12 }}>Which of these did you actually do?</h3>
                  {script.length === 0 && <p className="small">This was a clean take.</p>}
                  {script.map((s) => (
                    <label key={s.id} className="small" style={{ display: "flex", gap: 8, marginBottom: 6 }}>
                      <input type="checkbox" checked={s.done}
                             onChange={(e) => setScript(script.map((x) => x.id === s.id ? { ...x, done: e.target.checked } : x))} />
                      <span><b>Bar {s.bar}, beat {s.beatLabel}:</b> {s.text}</span>
                    </label>
                  ))}
                  <label className="small" style={{ display: "flex", gap: 8, margin: "8px 0" }}>
                    <input type="checkbox" checked={usable} onChange={(e) => setUsable(e.target.checked)} />
                    <span>Everything else went as written. Untick if you made other mistakes, lost your place, or something interrupted.</span>
                  </label>
                  <div className="row">
                    <button className="primary" onClick={() => submit(take)}>Save take</button>
                    <button className="danger" onClick={() => { setTake(null); setPhase("live"); setMsg("Take discarded."); }}>Discard</button>
                  </div>
                </>
              )}
            </div>
          )}
          <div className="panel">
            <h2>Live vision</h2>
            <p className="small muted">
              Blue = fretting hand, pink = strumming hand (decided by position along the neck, not by camera handedness).
              After calibration each fingertip is mapped to a string/fret cell through the fretboard homography.
            </p>
            <div className="row small">
              Target chord:
              <select value={preview} onChange={(e) => setPreview(e.target.value)} disabled={phase === "recording"}>
                {Object.keys(chords).map((c) => <option key={c}>{c}</option>)}
              </select>
            </div>
            {shape && (
              <div className="small cells" style={{ marginTop: 8 }}>
                {shape.frets.map((f, i) => f > 0 && (
                  <div key={i}>{FINGER_NAMES[shape.fingers[i]] ?? "?"} → string {6 - i}, fret {f}</div>
                ))}
              </div>
            )}
            <h3 style={{ marginTop: 12 }}>Fingertips now</h3>
            <div className="cells small">
              {!H && <div className="muted">Calibrate to map fingertips to cells.</div>}
              {H && tips.length === 0 && <div className="muted">No fretting hand in view.</div>}
              {H && tips.map((t) => (
                <div key={t.finger}>
                  {FINGER_NAMES[t.finger].padEnd(6)} {t.onNeck ? `string ${t.string}, fret ${t.fret}` : "off the neck"}
                </div>
              ))}
            </div>
          </div>
          <div className="panel small">
            <h3>Setup tips</h3>
            <ul style={{ paddingLeft: 18, margin: 0 }}>
              <li>Neck roughly horizontal, nut fully in frame, 0.6–1 m away, light from the front.</li>
              <li>A slightly top-down angle keeps fingertips from hiding behind the neck.</li>
              <li>Headphones for the click give the cleanest audio; speakers work too (latency is measured from the bleed).</li>
              <li>Hold the guitar still while calibrating; the server tracks the neck if it moves later.</li>
            </ul>
          </div>
        </aside>
      </div>
    </main>
  );
}
