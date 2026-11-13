# Recording test takes

These recordings tell us how accurate the grader really is on real guitars. You don't label anything by hand.
Before each take, the app tells you which mistakes to make on purpose. Afterwards you tick off the ones you
actually made, and those become the take's answer key.

## Setup (once)
1. Start the app (see the main README): backend on :8000, web on :3000.
2. Open **http://localhost:3000**. At the bottom, under *Recording test data?*, pick an exercise.
   You can also add `?collect=1` to any practice page URL.
3. Fill in the **Scripted take** panel: your name or initials, the guitar, the device, how you hear the click,
   and the room. The page remembers these, so you only change them when something changes.
4. Click **Start camera + mic**, set the camera up as described in *Setup tips*, and tap the 4 fretboard
   points (**Calibrate fretboard**).

## Each take (about 1 minute)
1. Read the **Mistakes to make this take** list. Usually 1–3 mistakes, and sometimes none, which means play it
   as cleanly as you can. If an instruction doesn't suit you, click **Deal a different script**.
2. Click **Record take** and play along to the click. A yellow cue warns you a bar before each mistake,
   and it turns red during the bar where you make it.
3. When it stops, tick only the mistakes you **actually** made. Leave **Everything else went as written**
   ticked only if the rest was clean. Untick it if you also fluffed something else, lost your place,
   or got interrupted.
4. Click **Save take**. A new script is dealt straight away for the next one.

Make each mistake clearly but naturally, the way a learner would. Do it **only on the strum named** and play
everything around it normally. A mistake you're unsure about is better left unticked than ticked.

## What to record
Aim for about **200 takes** in total. Even **20–30 takes** are enough for the first real accuracy numbers.

Vary things across takes rather than within one take:

| Vary | Values |
|---|---|
| Guitar | acoustic; electric through an amp; electro-acoustic |
| Device | laptop; phone; (tablet) |
| Click | speakers for most takes, some on headphones |
| Room | 2–3 different rooms (a bare room echoes, a bedroom doesn't) |
| Player | you plus a few friends, ideally different skill levels |
| Exercise | all of them, more takes of the ones with up-strums and chord changes |

**Keep the camera framing from the setup tips.** Re-calibrate (4 taps) whenever the camera or guitar moves
noticeably between takes.

## Optional: exact timing ground truth
The scripted "late" mistake only says that a strum was late, not by how many milliseconds. For about 20 takes
it's worth getting exact strum times. Record an electro-acoustic or electric guitar's line output at the same
time if you can, or mark the strum times by hand later with the marking tool once it's built. This is only
needed for millisecond-level timing numbers; everything else works without it.

## Getting the data out
Everything is saved under `api/data/sessions/<id>/`: the original recording, plus `labels.json` holding the
script, what you ticked, and your tags. To score the grader on all collected takes:

```bash
cd api
curl -s localhost:8000/api/collect/manifest > takes.json      # add ?include_unusable=true to include unticked takes
.venv/bin/python -m scripts.eval --manifest takes.json --by device,guitar,player,click_audio
```

To give the takes to someone else, zip the whole `api/data/` folder (it holds the recordings and the database).
They put it in place of their own `api/data/` and run the two commands above. `takes.json` contains file paths
from the machine it was exported on, so always re-export it on the machine that runs the eval.
