// Copies the MediaPipe WASM runtime out of node_modules and downloads the hand model,
// so the app serves them itself from /public (runs on npm install).
import { cpSync, existsSync, mkdirSync, writeFileSync } from "node:fs";

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";

cpSync("node_modules/@mediapipe/tasks-vision/wasm", "public/mediapipe/wasm", { recursive: true });
mkdirSync("public/models", { recursive: true });
if (!existsSync("public/models/hand_landmarker.task")) {
  const r = await fetch(MODEL_URL);
  if (!r.ok) throw new Error(`model download failed: ${r.status}`);
  writeFileSync("public/models/hand_landmarker.task", Buffer.from(await r.arrayBuffer()));
}
console.log("mediapipe assets ready");
