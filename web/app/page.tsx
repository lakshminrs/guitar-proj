"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type Exercise } from "@/lib/api";

export default function Home() {
  const [exs, setExs] = useState<Exercise[] | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => { api.exercises().then(setExs).catch((e) => setErr(String(e))); }, []);

  return (
    <main className="wrap">
      <h1>Pick an exercise</h1>
      <p className="muted">
        Play along to the click with your camera on the guitar neck. Afterwards you get a score, the top 3 things to fix,
        and a replay with an overlay showing which finger, string, fret and beat went wrong.
      </p>
      {err && <p style={{ color: "var(--bad)" }}>Can't reach the API ({err}). Is the backend running on :8000?</p>}
      <div className="grid cards">
        {exs?.map((e) => (
          <Link key={e.id} href={`/practice/${e.id}`} className="card">
            <h3>{e.title}</h3>
            <div className="muted small">{e.bpm} BPM · {e.time_sig} · {e.bars.length} bars</div>
            <div className="chips">
              {[...new Set(e.bars.map((b) => b.chord))].map((c) => <span className="chip" key={c}>{c}</span>)}
              <span className="chip">{e.bars[0].strum_pattern}</span>
            </div>
            <div className="small muted">{e.description}</div>
          </Link>
        ))}
      </div>
      {exs && exs.length > 0 && (
        <div className="panel small" style={{ marginTop: 24 }}>
          <h3>Recording test data?</h3>
          <p className="muted" style={{ marginTop: 0 }}>
            Scripted takes tell you which mistakes to make on purpose, so every recording comes with its own answer key.
            See <code>api/data/COLLECTION.md</code>.
          </p>
          <div className="chips">
            {exs.map((e) => <Link key={e.id} className="chip" href={`/practice/${e.id}?collect=1`}>{e.title}</Link>)}
          </div>
        </div>
      )}
    </main>
  );
}
