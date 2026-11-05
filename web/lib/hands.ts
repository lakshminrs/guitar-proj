import type { HandLandmarker } from "@mediapipe/tasks-vision";

let pending: Promise<HandLandmarker> | null = null;

/** MediaPipe HandLandmarker (WASM + GPU delegate), served from /public so it works offline. */
export function getHandLandmarker(): Promise<HandLandmarker> {
  if (!pending) {
    pending = (async () => {
      const { FilesetResolver, HandLandmarker } = await import("@mediapipe/tasks-vision");
      const fileset = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
      const opts = (delegate: "GPU" | "CPU") => ({
        baseOptions: { modelAssetPath: "/models/hand_landmarker.task", delegate },
        runningMode: "VIDEO" as const,
        numHands: 2,
        minHandDetectionConfidence: 0.4,
        minHandPresenceConfidence: 0.4,
        minTrackingConfidence: 0.4,
      });
      try {
        return await HandLandmarker.createFromOptions(fileset, opts("GPU"));
      } catch {
        return await HandLandmarker.createFromOptions(fileset, opts("CPU"));
      }
    })();
  }
  return pending;
}

export const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
