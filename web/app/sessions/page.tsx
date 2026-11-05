"use client";
import Link from "next/link";
import { useEffect, useState } from "react";
import { api, type Session } from "@/lib/api";

export default function Sessions() {
  const [rows, setRows] = useState<Session[] | null>(null);
  useEffect(() => { api.sessions().then(setRows).catch(() => setRows([])); }, []);
  return (
    <main className="wrap">
      <h1>Past takes</h1>
      {rows?.length === 0 && <p className="muted">No takes yet. <Link href="/">Pick an exercise</Link>.</p>}
      <div className="grid" style={{ gap: 8 }}>
        {rows?.map((s) => (
          <Link key={s.id} href={`/sessions/${s.id}`} className="card row" style={{ justifyContent: "space-between" }}>
            <span>Exercise #{s.exercise_id} · {new Date(s.created_at).toLocaleString()}</span>
            <span>{s.status === "done" ? <b>{s.score?.toFixed(0)}</b> : <span className="muted">{s.status}</span>}</span>
          </Link>
        ))}
      </div>
    </main>
  );
}
