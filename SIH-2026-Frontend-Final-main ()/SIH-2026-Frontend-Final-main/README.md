# THE SILENT DOG'S WHISTLE

## Running locally

```bash
npm install
npm run dev
```

The development server runs at `http://localhost:5174` (configured in `vite.config.ts`).

Backend API target (set in `.env`):

```bash
VITE_API_BASE_URL=http://127.0.0.1:8021
```

## Production build

```bash
npm run build
npm run preview
```

## Phase 1A

The home route includes a CSS-driven cinematic intro: darkness, two signal pulses, waveform, staged brand reveal, and a handoff into the minimal home hero. State and timing live in `src/context/IntroContext.tsx` and `src/components/intro/IntroSequence.tsx`.

`silentDogIntroSeen` is kept in `sessionStorage`. REPLAY INTRO resets the sequence without reloading. Reduced-motion users receive the static identity immediately. The CSS scene is deliberately the WebGL fallback; 3D/model loading is deferred to the next phase.

## Theme and future data integration

Palette tokens live in `src/index.css`. Existing typed demo fixtures remain in `src/data/mock.ts`; future API adapters should be added under `src/services/`.

## Future 3D model replacement

When Phase 1B begins, place the approved model at `public/models/silent-dog.glb` and load it through an isolated `SilentDogModel` component. No third-party 3D asset is bundled in this phase.
