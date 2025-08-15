export type ComprehensionScores = {
  benefit: number[];
  provisionQuality: number[];
  effort: number[];
};


/**
 * Compute constant-free directional comprehension scores from an attention matrix.
 * Matrix convention: attention[target][source].
 */
export function computeComprehension(
  attention: number[][],
  knownness?: number[]
): ComprehensionScores {
  const n = attention.length;
  if (n === 0) return { benefit: [], provisionQuality: [], effort: [] };
  const m = attention[0].length;
  if (n !== m) {
    throw new Error('Attention matrix must be square [n x n].');
  }

  const k = knownness && knownness.length === n ? knownness.slice() : new Array(n).fill(1);
  // positions: 0..n-1

  // ----- INTEGRATION (receiving from context) -----
  const leftIntegration = new Array(n).fill(0);
  const rightIntegration = new Array(n).fill(0);
  for (let t = 0; t < n; t++) {
    let lSum = 0;
    let rSum = 0;
    for (let s = 0; s < n; s++) {
      const a = attention[t][s] * k[s];
      if (s < t) {
        const dist = t - s;
        lSum += a / (1 + dist);
      } else if (s > t) {
        const dist = s - t;
        rSum += a / (1 + dist);
      }
    }
    leftIntegration[t] = lSum;
    rightIntegration[t] = rSum;
  }
  const integrationRaw = leftIntegration.map((l, i) => l + 0.5 * rightIntegration[i]);
  const maxInt = Math.max(0, ...integrationRaw);
  const integration = maxInt > 0 ? integrationRaw.map(v => v / maxInt) : new Array(n).fill(0);

  // ----- CONTRIBUTION (providing to context) -----
  const rightContrib = new Array(n).fill(0);
  const leftContrib = new Array(n).fill(0);
  for (let t = 0; t < n; t++) {
    let rSum = 0;
    let lSum = 0;
    for (let s = 0; s < n; s++) {
      const v = attention[t][s] * k[s];
      if (s > t) {
        const dist = s - t;
        rSum += v / (1 + dist);
      } else if (s < t) {
        const dist = t - s;
        lSum += v / (1 + dist);
      }
    }
    rightContrib[t] = rSum;
    leftContrib[t] = lSum;
  }
  const maxRC = Math.max(0, ...rightContrib);
  const maxLC = Math.max(0, ...leftContrib);
  const rightContribNorm = maxRC > 0 ? rightContrib.map(v => v / maxRC) : new Array(n).fill(0);
  const leftContribNorm = maxLC > 0 ? leftContrib.map(v => v / maxLC) : new Array(n).fill(0);
  const contributionRaw = rightContribNorm.map((v, i) => v - 0.5 * leftContribNorm[i]);
  const minCR = Math.min(0, ...contributionRaw);
  const maxCR = Math.max(0, ...contributionRaw);
  const contribution = maxCR > minCR
    ? contributionRaw.map(v => (v - minCR) / (maxCR - minCR))
    : new Array(n).fill(0);

  // ----- EFFORT -----
  const effortRaw = integration.map((intv, i) => (1 - intv) + (1 - contribution[i]));
  const maxEff = Math.max(0, ...effortRaw);
  const effort = maxEff > 0 ? effortRaw.map(v => v / maxEff) : new Array(n).fill(0);

  // Map to previous naming: benefit->integration, provisionQuality->contribution
  return { benefit: integration, provisionQuality: contribution, effort };
}

function maskAttentionByReadIndex(attention: number[][], readIndex: number): number[][] {
  const n = attention.length;
  const masked: number[][] = new Array(n);
  for (let t = 0; t < n; t++) {
    const row = attention[t];
    const out = new Array(n);
    for (let s = 0; s < n; s++) {
      // allow only sources up to the current read index
      out[s] = s <= readIndex ? row[s] : 0;
    }
    masked[t] = out;
  }
  return masked;
}

/**
 * Compute effort frames for each reading position (0..n-1),
 * limiting sources to those already "read" (<= index).
 */
export function computeEffortTimeline(
  attention: number[][],
  knownness?: number[]
): number[][] {
  const n = attention.length;
  if (n === 0) return [];
  const frames: number[][] = [];
  for (let i = 0; i < n; i++) {
    const masked = maskAttentionByReadIndex(attention, i);
    const { effort } = computeComprehension(masked, knownness);
    frames.push(effort);
  }
  return frames;
}
