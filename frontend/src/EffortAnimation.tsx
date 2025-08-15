import { useEffect, useMemo, useRef, useState } from 'react';

interface Props {
  words: string[];
  attention: number[][];
  knownness: number[];
}

import { computeEffortTimeline } from './utils/computeComprehension';

export default function EffortAnimation({ words, attention, knownness }: Props) {
  const frames = useMemo(() => computeEffortTimeline(attention, knownness), [attention, knownness]);
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const rafRef = useRef<number | null>(null);
  const lastTs = useRef<number | null>(null);

  // 3 frames per second progression by default
  const fps = 3;

  useEffect(() => {
    if (!playing) return;
    const step = (ts: number) => {
      if (lastTs.current == null) lastTs.current = ts;
      const dt = ts - lastTs.current;
      const interval = 1000 / fps;
      if (dt >= interval) {
        setIdx(prev => (prev + 1) % Math.max(1, frames.length));
        lastTs.current = ts;
      }
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      if (rafRef.current != null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      lastTs.current = null;
    };
  }, [playing, frames.length]);

  useEffect(() => {
    // reset when words/frames change
    setIdx(0);
  }, [frames.length]);

  if (!words.length || !frames.length) return null;

  const effort = frames[idx] ?? [];

  return (
    <div style={{margin: '1em 0'}}>
      <div style={{display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center'}}>
        <button onClick={() => setPlaying(p => !p)}>{playing ? 'Pause' : 'Play'}</button>
        <input
          type="range"
          min={0}
          max={Math.max(0, frames.length - 1)}
          value={idx}
          onChange={e => setIdx(Number(e.target.value))}
          style={{width: 240}}
        />
        <span style={{fontSize: '.9em', color: '#aaa'}}>Step {idx + 1}/{frames.length}</span>
      </div>
      <div style={{display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'center', marginTop: 8}}>
        {words.map((w, i) => {
          const e = Math.max(0, Math.min(1, effort[i] ?? 0));
          const bg = `oklch(${(35 + 45 * (1 - e)).toFixed(1)}% ${(.08 + .16 * e).toFixed(3)} 29)`; // blueish scale by effort
          const color = 'var(--color-text)';
          const border = '1px solid var(--color-border)';
          return (
            <span key={i} style={{
              padding: '4px 8px',
              borderRadius: 8,
              background: bg,
              color,
              border,
            }} title={`Effort ${(e*100).toFixed(1)}%`}>
              {w}
            </span>
          );
        })}
      </div>
    </div>
  );
}
