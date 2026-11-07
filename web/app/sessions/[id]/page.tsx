"use client";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Chord, type GradedEvent, type GradeError, type Patterns, type Session, type Track } from "@/lib/api";
import { COLORS, drawGrid, drawHand, fillCell, label, mapper, ring, strokeString } from "@/lib/draw";
import { applyH, FINGER_TIPS, type Mat3, type Pt } from "@/lib/geometry";

const COMP_LABEL: Record<string, string> = { fingering: "Fingering", pitch: "Pitch", timing: "Timing", direction: "Strum direction" };
const toPts = (a: number[] | null): Pt[] | null => (a ? Array.from({ length: a.length / 2 }, (_, i) => [a[2 * i], a[2 * i + 1]] as Pt) : null);

const DIFF_WORD = { missing: "missed", extra: "extra strum", direction: "wrong direction" } as const;

/** Played strum pattern per bar under the chart's pattern; mismatched slots highlighted. */
function PatternPanel({ p, onBar }: { p: Patterns; onBar: (bar: number) => void }) {
  const res = p.resolution;
  const cell = (ch: string, k: number, kind?: string) => (
    <span key={k} title={kind} style={{
      display: "inline-block", width: "1.3em", textAlign: "center", fontFamily: "ui-monospace, monospace",
      borderLeft: k % res === 0 ? "1px solid var(--line)" : undefined,
      color: kind ? "var(--bad)" : ch === "-" ? "var(--muted)" : undefined, fontWeight: kind ? 700 : undefined,
    }}>{ch === "-" ? "·" : ch === "x" ? "•" : ch}</span>
  );
  return (
    <div className="panel">
      <h3>Strum pattern <span className="muted small">
        {p.bars_matching}/{p.bars.length} bars as written · same rhythm in {Math.round((p.consistency ?? 0) * 100)}% of bars
        · • = strummed, direction unclear</span></h3>
      <div style={{ overflowX: "auto" }}>
        <table className="small" style={{ borderCollapse: "collapse" }}>
          <tbody>
            {p.bars.map((b) => {
              const bad = new Map(b.diffs.map((d) => [d.slot, d.kind]));
              return (
                <tr key={b.bar} onClick={() => onBar(b.bar)} style={{ cursor: "pointer" }}>
                  <td className="muted" style={{ paddingRight: 8 }}>bar {b.bar}</td>
                  <td style={{ paddingRight: 12 }}>{b.chord}</td>
                  <td style={{ paddingRight: 16 }}>
                    <div className="muted">{[...b.expected].map((ch, k) => cell(ch, k))}</div>
                    <div>{[...b.played].map((ch, k) => cell(ch, k, bad.get(k)))}</div>
                  </td>
                  <td>{b.matches ? <span style={{ color: "var(--good)" }}>✓</span>
                    : <span className="muted">{b.diffs.map((d) => `${d.beat_label}: ${DIFF_WORD[d.kind]}`).join(", ")}</span>}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="muted small" style={{ margin: "6px 0 0" }}>Top row: the chart. Bottom row: what you played. Click a bar to jump there.</p>
    </div>
  );
}

export default function Replay() {
  const { id } = useParams<{ id: string }>();
  const [s, setS] = useState<Session | null>(null);
  const [events, setEvents] = useState<GradedEvent[]>([]);
  const [track, setTrack] = useState<Track | null>(null);
  const [chords, setChords] = useState<Record<string, Chord>>({});
  const [active, setActive] = useState<GradeError | null>(null);
  const [now, setNow] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // poll until done
  useEffect(() => {
    let stop = false;
    const tick = async () => {
      try {
        const x = await api.session(id);
        if (stop) return;
        setS(x);
        if (x.status === "done") {
          const [ev, tr] = await Promise.all([api.events(id), api.track(id)]);
          setEvents(ev); setTrack(tr);
          return;
        }
        if (x.status !== "failed") setTimeout(tick, 1000);
      } catch { if (!stop) setTimeout(tick, 2000); }
    };
    tick();
    api.chords().then((cs) => setChords(Object.fromEntries(cs.map((c) => [c.name, c])))).catch(() => {});
    return () => { stop = true; };
  }, [id]);

  const errors = s?.errors ?? [];
  const playable = errors.filter((e) => e.type !== "finger_position" || e.confidence > 0.5);

  // overlay render loop
  const state = useRef({ events, track, chords, active });
  state.current = { events, track, chords, active };
  useEffect(() => {
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const v = videoRef.current, c = canvasRef.current;
      if (!v || !c || !v.videoWidth) return;
      if (c.width !== v.videoWidth) { c.width = v.videoWidth; c.height = v.videoHeight; }
      const g = c.getContext("2d")!;
      g.clearRect(0, 0, c.width, c.height);
      const { events, track, chords, active } = state.current;
      const tms = v.currentTime * 1000;
      setNow(tms);
      const m = mapper(c.width, c.height, false);
      const fr = track ? track.frames[Math.min(track.frames.length - 1, Math.floor((v.currentTime * track.fps) / track.every))] : null;
      const H = (fr?.H as Mat3 | null) ?? null;

      // the event whose strum we're at (shape shown for 400 ms around it)
      const ev = [...events].reverse().find((e) => e.t_ms <= tms + 60);
      const near = ev && tms - ev.t_ms < 450 ? ev : null;
      if (H) {
        drawGrid(g, H, m);
        const chord = near?.expected_chord ?? ev?.expected_chord;
        const shape = chord ? chords[chord] : null;
        shape?.frets.forEach((f, i) => { if (f > 0) fillCell(g, H, m, 6 - i, f, COLORS.target, COLORS.targetStroke); });
      }
      const fl = toPts(fr?.fret ?? null), sl = toPts(fr?.strum ?? null);
      if (sl) drawHand(g, sl, m, COLORS.strumHand);
      if (fl) drawHand(g, fl, m, COLORS.fretHand);

      // errors of the current event (or the one selected in the list)
      const errs = active && Math.abs(active.t_ms - tms) < 700 ? [active] : near?.errors ?? [];
      for (const e of errs) {
        for (const d of e.overlay?.draw ?? []) {
          if (d.kind === "cell" && H) fillCell(g, H, m, d.string, d.fret, "rgba(61,220,132,0.25)", COLORS.targetStroke);
          if (d.kind === "string" && H) strokeString(g, H, m, d.string, COLORS.error);
          if (d.kind === "fingertip" && fl) ring(g, m(fl[FINGER_TIPS[d.finger]]), COLORS.error, 16);
        }
      }
      if (errs.length) {
        g.save(); g.font = "600 22px system-ui"; g.fillStyle = "rgba(0,0,0,0.7)";
        const text = errs[0].message, w = Math.min(g.measureText(text).width + 24, c.width - 20);
        g.fillRect(10, c.height - 54, w, 40); g.fillStyle = "#fff"; g.fillText(text, 22, c.height - 26, w - 24); g.restore();
      }
      if (near && H) {
        const p = m(applyH(H, [0.62, -0.25]));
        label(g, p, `${near.expected_chord} · bar ${near.bar} beat ${near.beat_label} · ${near.expected_direction === "D" ? "↓" : "↑"}`);
      }
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, []);

  const seek = (e: GradeError) => {
    setActive(e);
    const v = videoRef.current;
    if (v) { v.currentTime = Math.max(0, e.t_ms / 1000 - 0.3); v.play(); window.setTimeout(() => v.pause(), 900); }
  };

  const timing = useMemo(() => events.filter((e) => e.timing_ms != null), [events]);
  const dur = events.length ? events[events.length - 1].t_ms + 1000 : 1;

  if (!s) return <main className="wrap"><p className="muted">Loading…</p></main>;
  if (s.status !== "done") {
    return (
      <main className="wrap">
        <h1>{s.status === "failed" ? "Analysis failed" : "Analyzing your take…"}</h1>
        <p className="muted">{s.progress ?? s.status}</p>
        {s.error && <pre className="panel small" style={{ whiteSpace: "pre-wrap" }}>{s.error}</pre>}
      </main>
    );
  }

  const meta = (s.meta ?? {}) as Record<string, unknown>;
  const vstats = meta.vision_stats as Record<string, number> | null;
  return (
    <main className="wrap">
      <div className="row" style={{ justifyContent: "space-between", marginBottom: 12 }}>
        <h1>Replay</h1>
        <Link href={`/practice/${s.exercise_id}`}>Try again →</Link>
      </div>
      <div className="layout">
        <div className="grid">
          <div className="stage">
            {s.video_url
              ? <video ref={videoRef} src={s.video_url} controls playsInline />
              : <div style={{ display: "grid", placeItems: "center", height: "100%" }}>
                  <audio ref={videoRef as unknown as React.RefObject<HTMLAudioElement>} src={s.audio_url ?? ""} controls />
                </div>}
            <canvas ref={canvasRef} style={{ pointerEvents: "none" }} />
          </div>

          <div className="panel">
            <h3>Timing vs click <span className="muted small">(each strum; up = late, down = early; band = ±20 ms)</span></h3>
            <svg viewBox={`0 0 1000 120`} style={{ width: "100%", height: 120 }} preserveAspectRatio="none">
              <rect x="0" y={60 - 20 * 0.4} width="1000" height={40 * 0.4} fill="rgba(61,220,132,0.15)" />
              <line x1="0" x2="1000" y1="60" y2="60" stroke="#555" />
              {timing.map((e) => {
                const x = (e.t_ms / dur) * 1000, dt = Math.max(-140, Math.min(140, e.timing_ms!));
                const bad = Math.abs(dt) > 40;
                return <line key={e.id} x1={x} x2={x} y1={60} y2={60 - dt * 0.4} stroke={bad ? "#ff4d4f" : "#3ddc84"} strokeWidth="4" />;
              })}
              {events.filter((e) => e.missed).map((e) => <circle key={e.id} cx={(e.t_ms / dur) * 1000} cy="60" r="5" fill="#ff4d4f" />)}
              <line x1={(now / dur) * 1000} x2={(now / dur) * 1000} y1="0" y2="120" stroke="#ffd666" />
            </svg>
          </div>

          {s.patterns && <PatternPanel p={s.patterns} onBar={(bar) => {
            const e = events.find((x) => x.bar === bar);
            const v = videoRef.current;
            if (e && v) { v.currentTime = Math.max(0, e.t_ms / 1000 - 0.3); v.play(); }
          }} />}
        </div>

        <aside className="grid">
          <div className="panel">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <div><div className="muted small">Take score</div><div className="score">{s.score?.toFixed(0)}</div></div>
              <div className="small muted" style={{ textAlign: "right" }}>
                {String(meta.n_matched)}/{String(meta.n_expected)} strums matched<br />
                {meta.grid ? `grid: ${String(meta.vision_source)}` : "audio-only (no grid)"}
              </div>
            </div>
            <div className="grid" style={{ gap: 8, marginTop: 12 }}>
              {Object.entries(s.components ?? {}).map(([k, v]) => (
                <div key={k}>
                  <div className="row small" style={{ justifyContent: "space-between" }}><span>{COMP_LABEL[k] ?? k}</span><span>{v == null ? "n/a" : v.toFixed(0)}</span></div>
                  <div className="bar"><i style={{ width: `${v ?? 0}%`, background: v == null ? "#444" : v > 80 ? "var(--good)" : v > 60 ? "var(--accent)" : "var(--bad)" }} /></div>
                </div>
              ))}
            </div>
          </div>

          <div className="panel">
            <h2>Fix these first</h2>
            {(s.top_errors ?? []).length === 0 && <p className="muted">Nothing major — clean take!</p>}
            <ol style={{ paddingLeft: 18, margin: 0 }} className="grid">
              {(s.top_errors ?? []).map((t, i) => (
                <li key={i}>
                  <button className="err" onClick={() => {
                    const e = errors.find((x) => x.t_ms === t.example.t_ms) ?? { ...t, ...t.example, overlay: undefined } as unknown as GradeError;
                    seek(e);
                  }}>
                    {t.message}<div className="muted small">e.g. bar {t.example.bar}, beat {t.example.beat}</div>
                  </button>
                </li>
              ))}
            </ol>
            {s.coaching_note && <p className="small" style={{ marginTop: 10 }}>{s.coaching_note}</p>}
          </div>

          <div className="panel">
            <h2>All issues <span className="muted small">({playable.length})</span></h2>
            <div className="errlist">
              {playable.map((e, i) => (
                <button key={i} className={`err ${active === e ? "active" : ""}`} onClick={() => seek(e)}>
                  <span className="tag">{e.type.replace(/_/g, " ")}</span>
                  <span className="muted small">bar {e.bar} · {(e.t_ms / 1000).toFixed(1)}s · conf {Math.round(e.confidence * 100)}%</span>
                  <div className="small">{e.message}</div>
                </button>
              ))}
            </div>
          </div>
          {vstats && (
            <div className="panel small muted">
              Vision: {vstats.frames_with_grid}/{vstats.frames} frames with grid · fretting hand in {vstats.frames_with_fret_hand} ·
              strumming hand in {vstats.frames_with_strum_hand}
            </div>
          )}
        </aside>
      </div>
    </main>
  );
}
