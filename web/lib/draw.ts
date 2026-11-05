import { applyH, canonToCell, cellQuad, FINGER_TIPS, FRET_X, invert, N_FRETS, STRING_Y, type Mat3, type Pt } from "./geometry";
import { HAND_CONNECTIONS } from "./hands";

/** maps normalized raw-video coords to canvas pixels (handles mirrored display) */
export type Mapper = (p: Pt) => Pt;
export const mapper = (w: number, h: number, mirrored: boolean): Mapper =>
  ([x, y]) => [(mirrored ? 1 - x : x) * w, y * h];

export const COLORS = {
  grid: "rgba(255,255,255,0.55)", fret: "rgba(255,214,102,0.9)", nut: "#ffd666",
  target: "rgba(61,220,132,0.45)", targetStroke: "#3ddc84", error: "#ff4d4f",
  fretHand: "#4dabf7", strumHand: "#f783ac",
};

export function drawGrid(ctx: CanvasRenderingContext2D, H: Mat3, m: Mapper, frets = 12) {
  ctx.save();
  ctx.lineWidth = 1.5;
  for (let n = 0; n <= Math.min(frets, N_FRETS); n++) {
    const a = m(applyH(H, [FRET_X[n], 0])), b = m(applyH(H, [FRET_X[n], 1]));
    ctx.strokeStyle = n === 0 ? COLORS.nut : COLORS.fret;
    ctx.lineWidth = n === 0 ? 4 : 1.5;
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
    if ([3, 5, 7, 9, 12].includes(n)) {
      const c = m(applyH(H, [(FRET_X[n - 1] + FRET_X[n]) / 2, 1.18]));
      ctx.fillStyle = COLORS.fret; ctx.font = "bold 12px system-ui"; ctx.textAlign = "center";
      ctx.fillText(String(n), c[0], c[1]);
    }
  }
  ctx.strokeStyle = COLORS.grid; ctx.lineWidth = 1;
  STRING_Y.forEach((y, i) => {
    const a = m(applyH(H, [0, y])), b = m(applyH(H, [FRET_X[Math.min(frets, N_FRETS)], y]));
    ctx.lineWidth = 0.8 + (5 - i) * 0.25;
    ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke();
  });
  ctx.restore();
}

export function fillCell(ctx: CanvasRenderingContext2D, H: Mat3, m: Mapper, string: number, fret: number, fill: string, stroke?: string) {
  const q = cellQuad(string, fret).map((p) => m(applyH(H, p)));
  ctx.save();
  ctx.beginPath(); q.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p))); ctx.closePath();
  ctx.fillStyle = fill; ctx.fill();
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 2; ctx.stroke(); }
  ctx.restore();
}

export function strokeString(ctx: CanvasRenderingContext2D, H: Mat3, m: Mapper, string: number, color: string) {
  const y = STRING_Y[6 - string];
  const a = m(applyH(H, [-0.03, y])), b = m(applyH(H, [FRET_X[12], y]));
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 4; ctx.globalAlpha = 0.85;
  ctx.beginPath(); ctx.moveTo(...a); ctx.lineTo(...b); ctx.stroke(); ctx.restore();
}

export function drawHand(ctx: CanvasRenderingContext2D, lm: Pt[], m: Mapper, color: string) {
  ctx.save();
  ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.globalAlpha = 0.9;
  for (const [a, b] of HAND_CONNECTIONS) {
    ctx.beginPath(); ctx.moveTo(...m(lm[a])); ctx.lineTo(...m(lm[b])); ctx.stroke();
  }
  ctx.fillStyle = color;
  lm.forEach((p, i) => {
    const [x, y] = m(p);
    ctx.beginPath(); ctx.arc(x, y, [4, 8, 12, 16, 20].includes(i) ? 5 : 3, 0, Math.PI * 2); ctx.fill();
  });
  ctx.restore();
}

export function label(ctx: CanvasRenderingContext2D, [x, y]: Pt, text: string, bg = "rgba(0,0,0,0.7)", fg = "#fff") {
  ctx.save();
  ctx.font = "600 12px system-ui";
  const w = ctx.measureText(text).width + 10;
  ctx.fillStyle = bg; ctx.fillRect(x - w / 2, y - 26, w, 18);
  ctx.fillStyle = fg; ctx.textAlign = "center"; ctx.fillText(text, x, y - 13);
  ctx.restore();
}

export function ring(ctx: CanvasRenderingContext2D, [x, y]: Pt, color: string, r = 11) {
  ctx.save(); ctx.strokeStyle = color; ctx.lineWidth = 3;
  ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
}

/** Fretting hand = the hand nearer the nut along the neck (not MediaPipe handedness). */
export function splitHands(hands: Pt[][], H: Mat3 | null): { fret: Pt[] | null; strum: Pt[] | null } {
  if (!hands.length || !H) return { fret: hands[0] ?? null, strum: hands[1] ?? null };
  const Hi = invert(H);
  const xs = hands.map((h) => applyH(Hi, [(h[0][0] + h[9][0]) / 2, (h[0][1] + h[9][1]) / 2])[0]);
  if (hands.length === 1) return xs[0] < 0.62 ? { fret: hands[0], strum: null } : { fret: null, strum: hands[0] };
  const order = xs.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
  return { fret: hands[order[0][1]], strum: hands[order[order.length - 1][1]] };
}

export function fingertipCells(lm: Pt[], H: Mat3) {
  const Hi = invert(H);
  return Object.entries(FINGER_TIPS).map(([f, idx]) => {
    const c = applyH(Hi, lm[idx]);
    const [s, fr] = canonToCell(c);
    const onNeck = c[0] > -0.05 && c[0] < FRET_X[N_FRETS] && c[1] > -0.25 && c[1] < 1.25;
    return { finger: Number(f), tip: lm[idx], string: s, fret: fr, onNeck };
  });
}
