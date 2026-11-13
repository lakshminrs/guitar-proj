/**
 * Scripted takes for building a labelled eval set: before recording, the app deals out a few
 * deliberate mistakes (bar, beat, what to do). Because the player was told what to do, the script
 * is the label and nothing needs annotating afterwards. After the take the player ticks off the
 * mistakes they actually pulled off, so flubbed instructions don't become wrong labels.
 */
import type { Chord, Exercise } from "./api";
import { FINGER_NAMES, STRING_NAMES } from "./geometry";

export type Label = { type: string; bar: number; beat: number; string?: number };
export type ScriptItem = { id: number; bar: number; beatLabel: string; text: string; label: Label; done: boolean };

type Slot = { bar: number; slot: number; sub: number; beat: number; dir: string; label: string; pos: number };

const beatLabel = (slot: number, sub: number) => {
  const b = Math.floor(slot / sub) + 1, r = slot % sub;
  return r === 0 ? String(b) : sub === 2 ? `${b}&` : `${b}.${r + 1}`;
};

function slots(ex: Exercise): Slot[] {
  const bpb = parseInt(ex.time_sig.split("/")[0], 10);
  const out: Slot[] = [];
  ex.bars.forEach((b, i) => {
    const pat = b.strum_pattern.replace(/ /g, "");
    const sub = pat.length / bpb;
    [...pat].forEach((c, s) => out.push({ bar: i + 1, slot: s, sub, beat: Math.floor(s / sub) + 1, dir: c, label: beatLabel(s, sub),
                                            pos: i * bpb + s / sub }));
  });
  return out;
}

const pick = <T,>(xs: T[]): T | undefined => xs[Math.floor(Math.random() * xs.length)];

/** 0-3 mistakes on different bars (never bar 1, so the player is settled), ~25% clean takes. */
export function makeScript(ex: Exercise, chords: Record<string, Chord>): ScriptItem[] {
  const all = slots(ex);
  const strums = all.filter((s) => s.dir === "D" || s.dir === "U");
  const spb = 60 / ex.bpm;
  const n = pick([0, 1, 1, 2, 2, 3])!;
  const bars = [...new Set(strums.map((s) => s.bar))].filter((b) => b > 1).sort(() => Math.random() - 0.5);
  const kinds = ["muted_string", "wrong_fret", "strum_too_wide", "wrong_direction", "missed_strum",
                 "late", "choked_chord", "extra_strum"].sort(() => Math.random() - 0.5);
  const out: ScriptItem[] = [];

  for (const kind of kinds) {
    if (out.length >= n) break;
    for (const bar of bars) {
      if (out.some((o) => o.bar === bar)) continue;
      const inBar = strums.filter((s) => s.bar === bar);
      const first = inBar[0];
      const ch = chords[ex.bars[bar - 1].chord];
      if (!first || !ch) continue;
      const item = build(kind, bar, first, inBar, all, ch, spb);
      if (item) { out.push({ id: out.length, done: true, ...item }); break; }
    }
  }
  return out.sort((a, b) => a.bar - b.bar);
}

function build(kind: string, bar: number, first: Slot, inBar: Slot[], all: Slot[], ch: Chord, spb: number):
    Omit<ScriptItem, "id" | "done"> | null {
  const at = (s: Slot) => ({ bar, beatLabel: s.label });
  const fretted = ch.frets.map((f, i) => ({ f, i, string: 6 - i, finger: ch.fingers[i] })).filter((x) => x.f > 0);
  const sounding = ch.frets.map((f, i) => ({ f, i, string: 6 - i })).filter((x) => x.f >= 0);
  switch (kind) {
    case "muted_string": {
      const s = pick(sounding);
      if (!s) return null;
      return { ...at(first), label: { type: "muted_string", bar, beat: first.beat, string: s.string },
        text: `Mute the ${STRING_NAMES[s.string]} string (string ${s.string}) on the first ${ch.name} strum — touch it lightly with a spare finger so it thuds. Play the rest of the bar normally.` };
    }
    case "wrong_fret": {
      const s = pick(fretted.filter((x) => x.f >= 2 && x.finger));
      if (!s) return null;
      return { ...at(first), label: { type: "wrong_fret", bar, beat: first.beat, string: s.string },
        text: `On the first ${ch.name} strum, put your ${FINGER_NAMES[s.finger]} finger one fret too low (fret ${s.f - 1} instead of ${s.f}, ${STRING_NAMES[s.string]} string). Fix it for the next strum.` };
    }
    case "strum_too_wide": {
      if (ch.frets[0] >= 0 || first.dir !== "D") return null;
      return { ...at(first), label: { type: "strum_too_wide", bar, beat: first.beat },
        text: `On the first ${ch.name} strum, hit ALL six strings — including the low E you'd normally skip.` };
    }
    case "wrong_direction": {
      const s = pick(inBar.filter((x) => x.dir === "U"));
      if (!s) return null;
      return { ...at(s), label: { type: "wrong_direction", bar, beat: s.beat },
        text: `On beat ${s.label} (an up-strum), strum DOWN instead. You'll need an extra up-swing without touching the strings before it.` };
    }
    case "missed_strum": {
      const s = pick(inBar.slice(1));
      if (!s) return null;
      return { ...at(s), label: { type: "missed_strum", bar, beat: s.beat },
        text: `Skip the strum on beat ${s.label}: keep your hand moving but miss the strings.` };
    }
    case "late": {
      if (!(first.slot === 0)) return null;
      return { ...at(first), label: { type: "late", bar, beat: first.beat },
        text: `Come in a little late on beat 1 — like you're scrambling to get to ${ch.name}. Just behind the click, not a whole half-beat.` };
    }
    case "choked_chord": {
      if (first.dir !== "D") return null;
      const idx = all.indexOf(first);
      const next = all.slice(idx + 1).find((x) => x.dir === "D" || x.dir === "U");
      const gap = next ? (next.pos - first.pos) * spb : 0;
      if (gap < 0.5) return null;
      return { ...at(first), label: { type: "choked_chord", bar, beat: first.beat },
        text: `Right after the first ${ch.name} strum, relax your fretting hand so the chord dies immediately (keep fingers on the strings, just release pressure). Press again for the next strum.` };
    }
    case "extra_strum": {
      // a rest slot; the grader files extra strums under the nearest expected strum (earlier on a tie)
      const rests = all.filter((x) => x.bar === bar && x.dir === "-");
      const r = pick(rests);
      if (!r) return null;
      const idx = all.indexOf(r);
      const prev = [...all.slice(0, idx)].reverse().find((x) => x.dir === "D" || x.dir === "U");
      const next = all.slice(idx + 1).find((x) => x.dir === "D" || x.dir === "U");
      const dist = (x?: Slot) => (x ? Math.abs(x.pos - r.pos) : Infinity);
      const near = dist(prev) <= dist(next) ? prev : next;
      if (!near) return null;
      return { bar, beatLabel: r.label, label: { type: "extra_strum", bar: near.bar, beat: near.beat },
        text: `Add an extra, full down-strum on beat ${r.label} (normally a rest).` };
    }
  }
  return null;
}
