// Fretboard geometry (mirror of api/app/analysis/geometry.py).
// Canonical space: x = distance from nut in units of scale length (12th fret = 0.5),
// y = 0 at the low-E edge, 1 at the high-E edge. Image space: normalized 0..1 raw video coords.

export type Pt = [number, number];
export type Mat3 = number[]; // row-major 9

export const N_FRETS = 15;
export const STRING_MARGIN = 0.08;
export const fretX = (n: number) => 1 - Math.pow(2, -n / 12);
export const FRET_X = Array.from({ length: N_FRETS + 1 }, (_, n) => fretX(n));
export const stringY = (idx: number) => STRING_MARGIN + (idx * (1 - 2 * STRING_MARGIN)) / 5; // idx 0 = low E
export const STRING_Y = Array.from({ length: 6 }, (_, i) => stringY(i));

export const CALIB_POINTS = ["nut_low", "nut_high", "f12_low", "f12_high"] as const;
export type CalibName = (typeof CALIB_POINTS)[number];
const CANON: Record<CalibName, Pt> = {
  nut_low: [0, 0], nut_high: [0, 1], f12_low: [0.5, 0], f12_high: [0.5, 1],
};

/** Solve the 4-point homography canonical -> image (DLT, 8x8 Gaussian elimination). */
export function homography(src: Pt[], dst: Pt[]): Mat3 | null {
  const A: number[][] = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i], [u, v] = dst[i];
    A.push([x, y, 1, 0, 0, 0, -u * x, -u * y, u]);
    A.push([0, 0, 0, x, y, 1, -v * x, -v * y, v]);
  }
  for (let c = 0; c < 8; c++) {
    let p = c;
    for (let r = c + 1; r < 8; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    if (Math.abs(A[p][c]) < 1e-12) return null;
    [A[c], A[p]] = [A[p], A[c]];
    for (let r = 0; r < 8; r++) {
      if (r === c) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k < 9; k++) A[r][k] -= f * A[c][k];
    }
  }
  const h = A.map((row, i) => row[8] / row[i]);
  return [...h, 1];
}

export function homographyFromCalib(points: Partial<Record<CalibName, Pt>>): Mat3 | null {
  if (!CALIB_POINTS.every((k) => points[k])) return null;
  return homography(CALIB_POINTS.map((k) => CANON[k]), CALIB_POINTS.map((k) => points[k]!));
}

export function applyH(H: Mat3, [x, y]: Pt): Pt {
  const w = H[6] * x + H[7] * y + H[8];
  return [(H[0] * x + H[1] * y + H[2]) / w, (H[3] * x + H[4] * y + H[5]) / w];
}

export function invert(m: Mat3): Mat3 {
  const [a, b, c, d, e, f, g, h, i] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [A, -(b * i - c * h), b * f - c * e, B, a * i - c * g, -(a * f - c * d), C, -(a * h - b * g), a * e - b * d].map(
    (v) => v / det,
  );
}

/** canonical point -> [string 1..6, fret]; fret 0 = open / behind the nut */
export function canonToCell([x, y]: Pt): [number, number] {
  let fret = 0;
  if (x > 0) {
    fret = FRET_X.findIndex((fx) => fx >= x);
    if (fret === -1) fret = N_FRETS + 1;
    fret = Math.max(1, fret);
  }
  let best = 0;
  STRING_Y.forEach((sy, i) => { if (Math.abs(sy - y) < Math.abs(STRING_Y[best] - y)) best = i; });
  return [6 - best, fret];
}

export function cellCenter(string: number, fret: number): Pt {
  const idx = 6 - string;
  const x = fret <= 0 ? -0.02 : FRET_X[fret - 1] + 0.75 * (FRET_X[fret] - FRET_X[fret - 1]);
  return [x, STRING_Y[idx]];
}

/** polygon of a cell in canonical space */
export function cellQuad(string: number, fret: number): Pt[] {
  const idx = 6 - string;
  const half = (1 - 2 * STRING_MARGIN) / 10;
  const y0 = STRING_Y[idx] - half, y1 = STRING_Y[idx] + half;
  const x0 = fret <= 0 ? -0.03 : FRET_X[fret - 1], x1 = fret <= 0 ? 0 : FRET_X[fret];
  return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
}

export const FINGER_TIPS: Record<number, number> = { 1: 8, 2: 12, 3: 16, 4: 20 };
export const FINGER_NAMES: Record<number, string> = { 1: "index", 2: "middle", 3: "ring", 4: "pinky" };
export const STRING_NAMES: Record<number, string> = { 6: "low E", 5: "A", 4: "D", 3: "G", 2: "B", 1: "high E" };
