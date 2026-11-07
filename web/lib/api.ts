export type Bar = { chord: string; strum_pattern: string };
export type Exercise = { id: number; title: string; description: string; bpm: number; time_sig: string; bars: Bar[] };
export type Chord = { id: number; name: string; frets: number[]; fingers: number[]; midi: number[]; barre: number };

export type DrawCmd =
  | { kind: "cell"; string: number; fret: number; style: string }
  | { kind: "fingertip"; finger: number; style: string }
  | { kind: "string"; string: number; style: string }
  | { kind: "timing"; offset_ms: number }
  | { kind: "strum"; expected: string; observed: string; style: string }
  | { kind: "beat" | "chord"; style: string; expected?: string; observed?: string };

export type GradeError = {
  t_ms: number; bar: number; beat: number; chord: string | null; type: string;
  string: number | null; finger: number | null; expected_fret: number | null; observed_fret: number | null;
  severity: number; confidence: number; message: string; offset_ms?: number;
  overlay?: { frame: number | null; draw: DrawCmd[] };
};

export type TopError = {
  type: string; chord: string | null; count: number; severity: number; confidence: number;
  message: string; example: { t_ms: number; bar: number; beat: number };
};

export type Session = {
  id: string; exercise_id: number; status: string; progress: string | null; error: string | null;
  score: number | null; created_at: string; video_url: string | null; audio_url: string | null;
  components: Record<string, number | null> | null; top_errors: TopError[] | null;
  coaching_note: string | null; errors?: GradeError[]; meta?: Record<string, unknown>;
  patterns?: Patterns | null;
};

export type PatternDiff = { slot: number; beat_label: string; expected: string; played: string; kind: "missing" | "extra" | "direction" };
export type Patterns = {
  resolution: number; bars_matching: number; match_rate: number | null; consistency: number | null;
  most_common_rhythm: string; diff_counts: Record<string, number>;
  bars: { bar: number; chord: string; expected: string; played: string; matches: boolean; diffs: PatternDiff[] }[];
};

export type GradedEvent = {
  id: number; t_ms: number; expected_t_ms: number; bar: number; beat: number; beat_label: string;
  expected_chord: string; expected_direction: string; observed_direction: string | null;
  timing_ms: number | null; missed: boolean; frame: number;
  scores: Record<string, number | null>; errors: GradeError[];
  fingertips: { canon: Record<string, number[]>; image: Record<string, number[]>; cells: Record<string, number[]> };
};

export type TrackFrame = { H: number[] | null; fret: number[] | null; strum: number[] | null };
export type Track = { fps: number; width: number; height: number; every: number; mirrored: boolean; source: string; frames: TrackFrame[] };

async function j<T>(r: Response): Promise<T> {
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json() as Promise<T>;
}

export const api = {
  exercises: () => fetch("/api/exercises").then((r) => j<Exercise[]>(r)),
  exercise: (id: number) => fetch(`/api/exercises/${id}`).then((r) => j<Exercise>(r)),
  chords: () => fetch("/api/chords").then((r) => j<Chord[]>(r)),
  sessions: () => fetch("/api/sessions").then((r) => j<Session[]>(r)),
  createSession: (exercise_id: number) =>
    fetch("/api/sessions", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ exercise_id }) })
      .then((r) => j<{ id: string; upload_url: string }>(r)),
  upload: (url: string, blob: Blob) =>
    fetch(url, { method: "PUT", headers: { "content-type": blob.type.split(";")[0] || "video/webm" }, body: blob }).then((r) => j(r)),
  analyze: (id: string, body: unknown) =>
    fetch(`/api/sessions/${id}/analyze`, { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(body) })
      .then((r) => j(r)),
  session: (id: string) => fetch(`/api/sessions/${id}`).then((r) => j<Session>(r)),
  events: (id: string) => fetch(`/api/sessions/${id}/events`).then((r) => j<GradedEvent[]>(r)),
  track: async (id: string) => {
    const r = await fetch(`/api/sessions/${id}/track`);
    return r.ok ? ((await r.json()) as Track) : null;
  },
};

export function expectedStrums(ex: Exercise) {
  const bpb = parseInt(ex.time_sig.split("/")[0], 10);
  const out: { bar: number; beatPos: number; chord: string; dir: string }[] = [];
  ex.bars.forEach((b, i) => {
    const pat = b.strum_pattern.replace(/ /g, "");
    const sub = pat.length / bpb;
    [...pat].forEach((c, s) => { if (c === "D" || c === "U") out.push({ bar: i + 1, beatPos: i * bpb + s / sub, chord: b.chord, dir: c }); });
  });
  return out;
}
