// Sample-accurate click track on the Web Audio clock. Every scheduled click time is logged
// (in AudioContext seconds) so the backend can build the beat grid. The click is a 3 kHz
// tone burst with a soft 2 ms attack: above the guitar's fundamentals and narrowband, so
// the analyzer can notch it out and use any mic bleed to measure latency.

export class Metronome {
  ctx: AudioContext;
  clickTimes: number[] = [];
  private timer: number | null = null;
  private next = 0;
  private beat = 0;

  constructor(ctx: AudioContext) { this.ctx = ctx; }

  private click(t: number, accent: boolean) {
    const o = this.ctx.createOscillator(), g = this.ctx.createGain();
    o.frequency.value = 3000;
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(accent ? 0.5 : 0.3, t + 0.002);
    g.gain.exponentialRampToValueAtTime(0.001, t + 0.03);
    o.connect(g).connect(this.ctx.destination);
    o.start(t);
    o.stop(t + 0.04);
  }

  /** Schedule `totalBeats` clicks starting at ctx time `startAt`. onBeat fires (approximately) per beat. */
  start(bpm: number, totalBeats: number, beatsPerBar: number, startAt: number, onBeat?: (beat: number) => void, onDone?: () => void) {
    const spb = 60 / bpm;
    this.next = startAt;
    this.beat = 0;
    this.clickTimes = [];
    const tick = () => {
      while (this.next < this.ctx.currentTime + 0.15 && this.beat < totalBeats) {
        this.click(this.next, this.beat % beatsPerBar === 0);
        this.clickTimes.push(this.next);
        const b = this.beat, at = this.next;
        window.setTimeout(() => onBeat?.(b), Math.max(0, (at - this.ctx.currentTime) * 1000));
        this.next += spb;
        this.beat++;
      }
      if (this.beat >= totalBeats) {
        window.setTimeout(() => onDone?.(), Math.max(0, (this.next - this.ctx.currentTime) * 1000));
        this.stop();
      }
    };
    tick();
    this.timer = window.setInterval(tick, 25);
  }

  stop() {
    if (this.timer !== null) window.clearInterval(this.timer);
    this.timer = null;
  }
}
