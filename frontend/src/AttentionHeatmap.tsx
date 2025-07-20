import React, { useState, useEffect } from 'react';

interface AttentionHeatmapProps {
  words: string[];
  attention: number[][]; // attention[i][j]: how much word j attends to word i
  normProvided?: number[]; // normalized total provided
  normReceived?: number[]; // normalized total received
  selectedTokenIndices?: number[]; // indices of selected tokens
  setSelectedTokenIndices?: (indices: number[]) => void; // setter
  unknownTokenIndices?: number[]; // indices of words marked as unknown
  knownTokenIndices?: number[]; // indices of words marked as known
  setUnknownTokenIndices?: (indices: number[]) => void; // setter for unknown words
  setKnownTokenIndices?: (indices: number[]) => void; // setter for known words
  punctuationIndices?: number[]; // indices of punctuation words
  includePunctuationInCalculations?: boolean; // whether punctuation is included in calculations
}

const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({ 
  words, 
  attention, 
  selectedTokenIndices = [], 
  setSelectedTokenIndices, 
  unknownTokenIndices = [], 
  knownTokenIndices = [],
  setUnknownTokenIndices,
  setKnownTokenIndices,
  punctuationIndices = [],
  includePunctuationInCalculations = true
}) => {
  const displayWords = words;
  // Adapt attention: unknown words only receive, not provide
  // Also, recalculate norm scores based on this modified attention
  const displayAttention = attention.map((row, i) =>
    unknownTokenIndices.includes(i)
      ? row.map(() => 0)
      : [...row]
  );

  // Recompute provided/received and norm scores based on displayAttention
  const provided = displayAttention.map(row => row.reduce((a, b) => a + b, 0));
  const received = words.map((_, i) => displayAttention.reduce((sum, row) => sum + row[i], 0));

  // Compute original min/max for received from the original attention matrix (before zeroing unknowns)
  const originalReceived = words.map((_, i) => attention.reduce((sum, row) => sum + row[i], 0));
  const originalMinReceived = Math.min(...originalReceived);
  const originalMaxReceived = Math.max(...originalReceived);

  // For normProvided, exclude unknowns from min/max and set their normProvided=0
  const nonUnknownIndices = words.map((_, i) => i).filter(i => !unknownTokenIndices.includes(i));
  const minProvided = nonUnknownIndices.length > 0 ? Math.min(...nonUnknownIndices.map(i => provided[i])) : 0;
  const maxProvided = nonUnknownIndices.length > 0 ? Math.max(...nonUnknownIndices.map(i => provided[i])) : 1;
  let normProvided = provided.map((v, i) =>
    unknownTokenIndices.includes(i)
      ? 0
      : (maxProvided - minProvided ? (v - minProvided) / (maxProvided - minProvided) : 0)
  );
  // For normReceived, use the original min/max (before zeroing unknowns)
  let normReceived = received.map(v => (originalMaxReceived - originalMinReceived ? (v - originalMinReceived) / (originalMaxReceived - originalMinReceived) : 0));

  // If a word is marked as known, set its normReceived to 1 and distribute its previous normReceived among its providers proportionally
  if (knownTokenIndices && knownTokenIndices.length > 0) {
    let normReceivedCopy = [...normReceived];
    let normProvidedCopy = [...normProvided];
    knownTokenIndices.forEach(iKnown => {
      // --- NormReceived logic (as before) ---
      const prevNormReceived = normReceivedCopy[iKnown];
      normReceivedCopy[iKnown] = 1;

      // --- NormProvided boost for providers ---
      // Find all providers (words that give attention to iKnown)
      const totalAttentionToKnown = attention.reduce((sum, row) => sum + row[iKnown], 0);
      const delta = 1 - prevNormReceived;
      if (totalAttentionToKnown > 0 && delta > 0) {
        attention.forEach((row, j) => {
          if (j !== iKnown && row[iKnown] > 0) {
            const prop = row[iKnown] / totalAttentionToKnown;
            normProvidedCopy[j] += prop * delta;
          }
        });
      }
    });
    normReceived = normReceivedCopy;
    normProvided = normProvidedCopy;
  }
  // For scoreboard and graph, reorder left/right/normDiff to match displayWords
  const leftAttentionScores = words.map((_, i) => {
    let score = 0;
    for (let j = 0; j < i; ++j) {
      score += displayAttention[i][j];
    }
    return score;
  });
  const rightAttentionScores = words.map((_, i) => {
    let score = 0;
    for (let j = i + 1; j < words.length; ++j) {
      score += displayAttention[i][j];
    }
    return score;
  });
  // Total attention received (sum of column)
  const totalReceived = received;
  // Total attention provided (sum of row)
  const totalProvided = provided;

  const [scoreSortMetric, setScoreSortMetric] = useState<'original' | 'left' | 'right' | 'received' | 'provided' | 'normSum'>('provided');
  // Percent-based band controls
  const [upperBandPercent, setUpperBandPercent] = useState(20); // percent of words in upper band
  const [lowerBandPercent, setLowerBandPercent] = useState(20); // percent of words in lower band
  // These are percent values (0-100)
  // The thresholds will be computed from the current metric distribution
  const [upperThreshold, setUpperThreshold] = useState(0.6); // computed, not user-editable
  const [lowerThreshold, setLowerThreshold] = useState(0.4); // computed, not user-editable
  
  // Band expansion state
  const [expandedBands, setExpandedBands] = useState<{[key: string]: boolean}>({
    lower: false,
    middle: false, 
    upper: false
  });

  // Use unique ids for each word position
  const wordObjs = displayWords.map((word, idx) => ({ word, index: idx }));
  // Compute metrics for each word position ONCE, in original order
  const metrics = wordObjs.map(({ word, index }) => ({
    word,
    index,
    left: leftAttentionScores[index],
    right: rightAttentionScores[index],
    received: totalReceived[index],
    provided: totalProvided[index],
    normProvided: normProvided[index],
    normReceived: normReceived[index],
    normSum: normProvided[index] + normReceived[index],
    isUnknown: unknownTokenIndices.includes(index),
    isKnown: knownTokenIndices.includes(index),
  }));
  // Sorting logic: sort metrics, do not recalculate
  // Filter out punctuation from scoreboard when not included in calculations
  const filteredMetrics = metrics.filter((_, i) => 
    !(punctuationIndices.includes(i) && !includePunctuationInCalculations)
  );
  const scoreboard = [...filteredMetrics];
  if (scoreSortMetric === 'original') {
    scoreboard.sort((a, b) => a.index - b.index);
  } else {
    scoreboard.sort((a, b) => a[scoreSortMetric] - b[scoreSortMetric]);
  }

  // Text score: average (mean) of normSum for all words (excluding punctuation when not included)
  const textScore = filteredMetrics.length > 0 ? filteredMetrics.reduce((sum, m) => sum + m.normSum, 0) / filteredMetrics.length : 0;

  // Suggestion: find the words with the lowest normalized difference
  // const worstDescribed = scoreboard.slice(-3).map(s => s.word); // bottom 3

  // Removed selectedWordIdx and selectedWordIdx2 (pick word functionality removed)

  // Color scale for score highlighting
  function getColorScale(arr: number[]) {
    const min = Math.min(...arr);
    const max = Math.max(...arr);
    return (v: number) => {
      const t = (v - min) / (max - min + 1e-8);
      // Red (low) to white (high)
      const r = 255;
      const g = Math.round(255 * t);
      const b = Math.round(255 * t);
      return `rgb(${r},${g},${b})`;
    };
  }
  // --- Multi-token selection coloring logic ---
  // If tokens are selected, color by attention to/from selected tokens depending on sorting
  let customColorArr: number[] | null = null;
  let customColorLabel = '';
  if (selectedTokenIndices && selectedTokenIndices.length > 0) {
    if (scoreSortMetric === 'provided') {
      // Color by attention given to selected tokens (row to selected columns)
      customColorArr = metrics.map((_, i) =>
        selectedTokenIndices.reduce((sum, selIdx) => sum + displayAttention[i][selIdx], 0)
      );
      customColorLabel = 'attention given to selected';
    } else if (scoreSortMetric === 'received') {
      // Color by attention received from selected tokens (column from selected rows)
      customColorArr = metrics.map((_, i) =>
        selectedTokenIndices.reduce((sum, selIdx) => sum + displayAttention[selIdx][i], 0)
      );
      customColorLabel = 'attention received from selected';
    } else {
      // Default: color by attention given to selected
      customColorArr = metrics.map((_, i) =>
        selectedTokenIndices.reduce((sum, selIdx) => sum + displayAttention[i][selIdx], 0)
      );
      customColorLabel = 'attention given to selected';
    }
  }

  // Always display original text in original order, colored by the current metric or selection
  const originalTextColorArr = customColorArr || metrics.map(m => {
    if (scoreSortMetric === 'original') return m.normProvided;
    if (scoreSortMetric === 'received') return m.normReceived;
    if (scoreSortMetric === 'provided') return m.normProvided;
    if (scoreSortMetric === 'normSum') return m.normSum;
    return m[scoreSortMetric]; // for 'left' and 'right'
  });
  const originalTextColorScale = getColorScale(originalTextColorArr);

  // Option to color by band
  const [colorByBand, setColorByBand] = useState(false);
  // Band colors: above=white, between=light red, below=red
  const bandColors = {
    above: '#fff', // white
    between: '#ffd6d6', // light red (middle)
    below: '#ff4d4d', // strong red
  };

  // Compute thresholds so that upperBandPercent% of words are above upperThreshold, and lowerBandPercent% below lowerThreshold
  function computePercentile(arr: number[], percent: number) {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = Math.max(0, Math.min(sorted.length - 1, Math.round((1 - percent / 100) * (sorted.length - 1))));
    return sorted[idx];
  }
  function computeLowerPercentile(arr: number[], percent: number) {
    if (arr.length === 0) return 0;
    const sorted = [...arr].sort((a, b) => a - b);
    const idx = Math.max(0, Math.min(sorted.length - 1, Math.round((percent / 100) * (sorted.length - 1))));
    return sorted[idx];
  }
  useEffect(() => {
    // Create filtered color array that matches scoreboard filtering
    const filteredColorArr = originalTextColorArr.filter((_, i) => 
      !(punctuationIndices.includes(i) && !includePunctuationInCalculations)
    );
    // Compute thresholds from filtered color metric to match scoreboard
    setUpperThreshold(computePercentile(filteredColorArr, upperBandPercent));
    setLowerThreshold(computeLowerPercentile(filteredColorArr, lowerBandPercent));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [originalTextColorArr, upperBandPercent, lowerBandPercent, punctuationIndices, includePunctuationInCalculations]);

  // State to show/hide the Word Attention Heatmap
  const [showWordHeatmap, setShowWordHeatmap] = useState(false);
  const [showFormulas, setShowFormulas] = useState(false);

  // Red-to-white color scale for heatmap
  function getRedWhiteColor(value: number, min: number, max: number) {
    const t = (value - min) / (max - min + 1e-8);
    const r = 255;
    const g = Math.round(255 * t);
    const b = Math.round(255 * t);
    return `rgb(${r},${g},${b})`;
  }

  // Compute min/max for attention values for color scaling
  const flatAttention = displayAttention.flat();
  
  // Use safe min/max calculation to avoid stack overflow with large arrays
  let attMin = Infinity;
  let attMax = -Infinity;
  for (const value of flatAttention) {
    if (value < attMin) attMin = value;
    if (value > attMax) attMax = value;
  }
  
  // Fallback values if arrays are empty
  if (!isFinite(attMin)) attMin = 0;
  if (!isFinite(attMax)) attMax = 1;

  return (
    <div style={{ overflowX: 'auto', margin: '1em 0' }}>
      {/* Move original text display above the line graph */}
      <div style={{margin: '16px 0', fontSize: '1.1em', lineHeight: 1.7}}>
        <div style={{display: 'flex', alignItems: 'center', gap: 16, marginBottom: 8}}>
          <b>Original Text{selectedTokenIndices && selectedTokenIndices.length > 0 ? ' (colored by ' + customColorLabel + ')' : ' (colored by ' + (scoreSortMetric === 'original' ? 'provided' : scoreSortMetric) + ')'}:</b>
          <div style={{fontSize: '0.8em'}}>
            <label style={{marginRight: 8}}>Sort by:</label>
            <select value={scoreSortMetric} onChange={e => setScoreSortMetric(e.target.value as 'original' | 'left' | 'right' | 'received' | 'provided' | 'normSum')} style={{fontSize: 13, padding: '2px 6px'}}>
              <option value="original">Norm. Sum (default)</option>
              <option value="left">Total Left</option>
              <option value="right">Total Right</option>
              <option value="received">Norm. Received</option>
              <option value="provided">Norm. Provided</option>
              <option value="normSum">Norm. Sum</option>
            </select>
          </div>
        </div>
        
        {/* Percent-based band controls and color by band toggle */}
        <div style={{margin: '8px 0', fontSize: '0.9em', display: 'flex', alignItems: 'center', gap: 16}}>
          <div>
            <label style={{marginRight: 8}}>Upper band (% of words):</label>
            <input
              type="number"
              value={upperBandPercent}
              onChange={e => setUpperBandPercent(Math.max(0, Math.min(100, Number(e.target.value))))}
              step="1"
              min="0"
              max="100"
              style={{width: '60px', padding: '2px 4px', marginRight: 16}}
            />
            <label style={{marginRight: 8}}>Lower band (% of words):</label>
            <input
              type="number"
              value={lowerBandPercent}
              onChange={e => setLowerBandPercent(Math.max(0, Math.min(100, Number(e.target.value))))}
              step="1"
              min="0"
              max="100"
              style={{width: '60px', padding: '2px 4px', marginRight: 16}}
            />
            <span style={{marginLeft: 12, color: '#888', fontSize: '0.95em'}}>
              (Current thresholds: upper ≥ {upperThreshold.toFixed(3)}, lower ≤ {lowerThreshold.toFixed(3)})
            </span>
          </div>
          <label style={{marginLeft: 24, display: 'flex', alignItems: 'center', gap: 4}}>
            <input
              type="checkbox"
              checked={colorByBand}
              onChange={e => setColorByBand(e.target.checked)}
              style={{marginRight: 4}}
            />
            Color by band
          </label>
        </div>

        <div style={{marginTop: 8, background: '#f8f8f8', padding: 20, borderRadius: 6, fontFamily: 'monospace', wordBreak: 'normal', position: 'relative', overflow: 'auto', minWidth: '1100px'}}>
          {/* Create a grid layout where each word maintains its horizontal position but moves vertically */}
          <div style={{position: 'relative', minHeight: '20em', width: '100%', lineHeight: '1.8', minWidth: '1100px'}}> {/* Increased height and width for line wrapping */}
            {/* Background color bands for all lines and threshold regions */}
            {(() => {
              // Use a single constant for band height and vertical spacing
              const BAND_HEIGHT_EM = 2.5;
              const ROWS_PER_LINE = 3; // above, between, below
              const LINE_GAP_EM = 4; // add extra gap between triple bands
              const wordWidths = metrics.map(met => (met.word.length * 9) + 14);
              const boxGap = 8;
              let currentLineWidth = 150;
              let lineNumber = 0;
              let maxLineNumber = 0;
              for (let i = 0; i < metrics.length; i++) {
                if (currentLineWidth + wordWidths[i] > 1000) {
                  lineNumber++;
                  currentLineWidth = 150;
                }
                currentLineWidth += wordWidths[i] + boxGap;
                maxLineNumber = Math.max(maxLineNumber, lineNumber);
              }
              // For each line and each band (above, between, below), render a band
              const bands = [];
              for (let l = 0; l <= maxLineNumber; l++) {
                // Each band is BAND_HEIGHT_EM tall, offset by l * (BAND_HEIGHT_EM * ROWS_PER_LINE + LINE_GAP_EM)
                // But the gap is only between the triple bands, not after each band
                const lineOffsetEm = l * (BAND_HEIGHT_EM * ROWS_PER_LINE + LINE_GAP_EM);
                bands.push(
                  <React.Fragment key={l}>
                    {/* Above threshold (top row) outline */}
                    <div style={{
                      position: 'absolute',
                      left: 0,
                      width: '100%',
                      top: `${0 + lineOffsetEm}em`,
                      height: `${BAND_HEIGHT_EM}em`,
                      background: 'none',
                      border: '1.5px solid #bbb',
                      borderRadius: '6px 6px 0 0',
                      zIndex: 10,
                      boxSizing: 'border-box',
                      pointerEvents: 'none',
                    }} />
                    {/* Between thresholds (middle row) outline */}
                    <div style={{
                      position: 'absolute',
                      left: 0,
                      width: '100%',
                      top: `${BAND_HEIGHT_EM + lineOffsetEm}em`,
                      height: `${BAND_HEIGHT_EM}em`,
                      background: 'none',
                      borderLeft: '1.5px solid #bbb',
                      borderRight: '1.5px solid #bbb',
                      zIndex: 10,
                      boxSizing: 'border-box',
                      pointerEvents: 'none',
                    }} />
                    {/* Below threshold (bottom row) outline */}
                    <div style={{
                      position: 'absolute',
                      left: 0,
                      width: '100%',
                      top: `${2 * BAND_HEIGHT_EM + lineOffsetEm}em`,
                      height: `${BAND_HEIGHT_EM}em`,
                      background: 'none',
                      border: '1.5px solid #bbb',
                      borderRadius: '0 0 6px 6px',
                      zIndex: 10,
                      boxSizing: 'border-box',
                      pointerEvents: 'none',
                    }} />
                  </React.Fragment>
                );
              }
              return bands;
            })()}
            {/* Removed left-side band labels to prevent overlap with words */}
            
            {/* Position each word in its original horizontal position but appropriate vertical level with line breaks */}
            {metrics.map((m, i) => {
              // ...existing code...
              const BAND_HEIGHT_EM = 2.5;
              const ROWS_PER_LINE = 3;
              const LINE_GAP_EM = 4;
              let valueForBand = originalTextColorArr[i];
              let verticalPosition = 1; // Default to middle band
              
              // Special handling for punctuation when not included in calculations
              if (punctuationIndices.includes(i) && !includePunctuationInCalculations) {
                verticalPosition = 1; // Force punctuation to middle band position
              } else {
                // For non-punctuation words, use the value from the filtered array for band determination
                const filteredColorArr = originalTextColorArr.filter((_, idx) => 
                  !(punctuationIndices.includes(idx) && !includePunctuationInCalculations)
                );
                const filteredIndex = originalTextColorArr.slice(0, i + 1)
                  .filter((_, idx) => !(punctuationIndices.includes(idx) && !includePunctuationInCalculations))
                  .length - 1;
                
                const filteredValue = filteredColorArr[filteredIndex];
                
                // Normal position determination based on filtered value and thresholds
                if (filteredValue >= upperThreshold) {
                  verticalPosition = 0;
                } else if (filteredValue <= lowerThreshold) {
                  verticalPosition = 2;
                }
              }
              const wordWidths = metrics.map(met => (met.word.length * 9) + 14);
              const boxGap = 8;
              let currentLineWidth = 150;
              let lineNumber = 0;
              for (let j = 0; j < i; j++) {
                if (currentLineWidth + wordWidths[j] > 1000) {
                  lineNumber++;
                  currentLineWidth = 150;
                }
                currentLineWidth += wordWidths[j] + boxGap;
              }
              if (currentLineWidth + wordWidths[i] > 1000) {
                lineNumber++;
                currentLineWidth = 150;
              }
              const estimatedLeftOffset = currentLineWidth;
              const topEm = verticalPosition * BAND_HEIGHT_EM + lineNumber * (BAND_HEIGHT_EM * ROWS_PER_LINE + LINE_GAP_EM);
              // Determine band for coloring
              let band: 'above' | 'between' | 'below' = 'between';
              
              // Special handling for punctuation when not included in calculations
              if (punctuationIndices.includes(i) && !includePunctuationInCalculations) {
                band = 'between'; // Force punctuation to middle band
              } else {
                // For non-punctuation words, use the same filtered array logic as positioning
                const filteredColorArr = originalTextColorArr.filter((_, idx) => 
                  !(punctuationIndices.includes(idx) && !includePunctuationInCalculations)
                );
                const filteredIndex = originalTextColorArr.slice(0, i + 1)
                  .filter((_, idx) => !(punctuationIndices.includes(idx) && !includePunctuationInCalculations))
                  .length - 1;
                
                const filteredValue = filteredColorArr[filteredIndex];
                
                // Normal band determination based on filtered value and thresholds
                if (filteredValue >= upperThreshold) band = 'above';
                else if (filteredValue <= lowerThreshold) band = 'below';
              }
              return (
                <span
                  key={m.index}
                  style={{
                    position: 'absolute',
                    left: `${estimatedLeftOffset}px`,
                    top: `${topEm}em`,
                    background: (punctuationIndices.includes(i) && !includePunctuationInCalculations)
                      ? '#e0e0e0' // Grey background for punctuation when not included in calculations
                      : colorByBand
                        ? bandColors[band]
                        : originalTextColorScale(originalTextColorArr[i]),
                    color: m.isUnknown ? '#b00' : '#000',
                    borderRadius: 4,
                    width: `${wordWidths[i]}px`,
                    minWidth: `${wordWidths[i]}px`,
                    maxWidth: `${wordWidths[i]}px`,
                    textAlign: 'center',
                    padding: '3px 0',
                    display: 'inline-block',
                    fontWeight: 500,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    cursor: 'pointer',
                    border: selectedTokenIndices.includes(i) ? '2px solid #0072B2' : '2px solid transparent',
                    boxShadow: selectedTokenIndices.includes(i) ? '0 0 4px #0072B2' : undefined,
                    transition: 'border 0.1s, box-shadow 0.1s, top 0.3s ease',
                    textDecoration: m.isUnknown ? 'underline wavy #b00' : undefined,
                    zIndex: 2,
                    marginRight: 0,
                    whiteSpace: 'nowrap',
                  }}
                  onClick={() => {
                    if (!setSelectedTokenIndices) return;
                    if (selectedTokenIndices.includes(i)) {
                      setSelectedTokenIndices(selectedTokenIndices.filter(idx => idx !== i));
                    } else {
                      setSelectedTokenIndices([...selectedTokenIndices, i]);
                    }
                  }}
                  title={
                    `Norm sum: ${m.normSum.toFixed(3)} (${
                      valueForBand > upperThreshold ? 'above threshold' :
                      valueForBand < lowerThreshold ? 'below threshold' :
                      'within thresholds'
                    })` + 
                    (m.isUnknown ? ' | Unknown word: only receives attention, does not provide.' : '') +
                    (selectedTokenIndices.length > 0
                      ? scoreSortMetric === 'provided'
                        ? ` | Attention given to selected: ${customColorArr ? customColorArr[i].toFixed(3) : ''}`
                        : scoreSortMetric === 'received'
                          ? ` | Attention received from selected: ${customColorArr ? customColorArr[i].toFixed(3) : ''}`
                          : ` | Attention given to selected: ${customColorArr ? customColorArr[i].toFixed(3) : ''}`
                      : scoreSortMetric === 'provided'
                        ? ` | Norm. provided: ${metrics[i].normProvided.toFixed(3)}`
                        : scoreSortMetric === 'received'
                          ? ` | Norm. received: ${metrics[i].normReceived.toFixed(3)}`
                          : '')
                  }
                >
                  {m.word}
                </span>
              );
            })}
          </div>
        </div>
        {/* Removed ScoreLineGraph component */}
      </div>
      {/* Scoreboard Table: Split by bands with unknown/known marking */}
      <div style={{ width: '100%' }}>
        <b>Scoreboard: Normalized Scores by Attention Bands | Text Score (mean norm sum per word): {textScore.toFixed(3)}</b>
        
        {/* Helper function to determine band for a word */}
        {(() => {
          // Create the same filtered data that was used for threshold calculation
          const filteredColorArr = originalTextColorArr.filter((_, i) => 
            !(punctuationIndices.includes(i) && !includePunctuationInCalculations)
          );
          
          // Get the same metric values that were used to calculate thresholds
          const getMetricValueForBanding = (item: typeof scoreboard[0]) => {
            const index = item.index;
            // Find the position of this word in the filtered array
            const filteredIndex = originalTextColorArr.slice(0, index + 1)
              .filter((_, i) => !(punctuationIndices.includes(i) && !includePunctuationInCalculations))
              .length - 1;
            
            // Use the value from the same filtered array used for thresholds
            return filteredColorArr[filteredIndex];
          };

          const upperBandWords = scoreboard.filter(item => {
            const value = getMetricValueForBanding(item);
            return value >= upperThreshold;
          });
          const middleBandWords = scoreboard.filter(item => {
            const value = getMetricValueForBanding(item);
            return value > lowerThreshold && value < upperThreshold;
          });
          const lowerBandWords = scoreboard.filter(item => {
            const value = getMetricValueForBanding(item);
            return value <= lowerThreshold;
          });

          const renderBandTable = (bandWords: typeof scoreboard, bandName: string, bandColor: string, bandKey: string) => {
            if (bandWords.length === 0) return null;
            
            const isExpanded = expandedBands[bandKey];
            const defaultLimit = 5; // Show 5 words by default
            const displayWords = isExpanded ? bandWords : bandWords.slice(0, defaultLimit);
            const hasMore = bandWords.length > defaultLimit;
            
            return (
              <div key={bandName} style={{ marginBottom: 20 }}>
                <h4 style={{ 
                  margin: '12px 0 6px 0', 
                  padding: '6px 12px', 
                  backgroundColor: bandColor, 
                  borderRadius: '4px',
                  fontSize: '0.9em',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center'
                }}>
                  <span>{bandName} Attention Band ({bandWords.length} words)</span>
                  {hasMore && (
                    <button
                      onClick={() => setExpandedBands(prev => ({
                        ...prev,
                        [bandKey]: !prev[bandKey]
                      }))}
                      style={{
                        fontSize: '12px',
                        padding: '2px 6px',
                        backgroundColor: 'rgba(0,0,0,0.1)',
                        border: '1px solid rgba(0,0,0,0.2)',
                        borderRadius: '3px',
                        cursor: 'pointer'
                      }}
                    >
                      {isExpanded ? `Show less (${defaultLimit})` : `Show all (${bandWords.length})`}
                    </button>
                  )}
                </h4>
                <table style={{borderCollapse: 'collapse', fontSize: '0.95em', width: '100%', border: `2px solid ${bandColor}`, tableLayout: 'fixed'}}>
                  <thead>
                    <tr style={{ backgroundColor: bandColor }}>
                      <th style={{textAlign: 'left', padding: '4px 8px', width: '25%'}}>Word</th>
                      <th style={{textAlign: 'right', padding: '4px 8px', width: '20%'}}>Norm. Received</th>
                      <th style={{textAlign: 'right', padding: '4px 8px', width: '20%'}}>Norm. Provided</th>
                      <th style={{textAlign: 'right', padding: '4px 8px', width: '20%'}}>Norm. Sum</th>
                      <th style={{textAlign: 'center', padding: '4px 8px', width: '15%'}}>Mark As</th>
                    </tr>
                  </thead>
                  <tbody>
                    {displayWords.map(({ word, normProvided, normReceived, index }) => (
                      <tr key={index}>
                        <td style={{padding: '4px 8px', fontWeight: 500, textAlign: 'left', width: '25%', wordWrap: 'break-word'}}>{word}</td>
                        <td style={{padding: '4px 8px', textAlign: 'right', width: '20%'}}>{normReceived.toFixed(3)}</td>
                        <td style={{padding: '4px 8px', textAlign: 'right', width: '20%'}}>{normProvided.toFixed(3)}</td>
                        <td style={{padding: '4px 8px', textAlign: 'right', width: '20%'}}>{(normProvided + normReceived).toFixed(3)}</td>
                        <td style={{padding: '4px 8px', textAlign: 'center', width: '15%'}}>
                          <div style={{ display: 'flex', gap: '4px', justifyContent: 'center' }}>
                            <button
                              onClick={() => {
                                if (!setUnknownTokenIndices || !setKnownTokenIndices) return;
                                if (unknownTokenIndices.includes(index)) {
                                  setUnknownTokenIndices(unknownTokenIndices.filter(i => i !== index));
                                } else {
                                  setUnknownTokenIndices([...unknownTokenIndices.filter(i => i !== index), index]);
                                  setKnownTokenIndices(knownTokenIndices.filter(i => i !== index));
                                }
                              }}
                              style={{
                                padding: '2px 6px',
                                fontSize: '11px',
                                backgroundColor: unknownTokenIndices.includes(index) ? '#ffcccc' : '#f8f8f8',
                                border: unknownTokenIndices.includes(index) ? '1px solid #ff6666' : '1px solid #ddd',
                                borderRadius: '3px',
                                cursor: 'pointer',
                                fontWeight: unknownTokenIndices.includes(index) ? 'bold' : 'normal'
                              }}
                              title="Mark as unknown word"
                            >
                              ?
                            </button>
                            <button
                              onClick={() => {
                                if (!setKnownTokenIndices || !setUnknownTokenIndices) return;
                                if (knownTokenIndices.includes(index)) {
                                  setKnownTokenIndices(knownTokenIndices.filter(i => i !== index));
                                } else {
                                  setKnownTokenIndices([...knownTokenIndices.filter(i => i !== index), index]);
                                  setUnknownTokenIndices(unknownTokenIndices.filter(i => i !== index));
                                }
                              }}
                              style={{
                                padding: '2px 6px',
                                fontSize: '11px',
                                backgroundColor: knownTokenIndices.includes(index) ? '#ccffcc' : '#f8f8f8',
                                border: knownTokenIndices.includes(index) ? '1px solid #66ff66' : '1px solid #ddd',
                                borderRadius: '3px',
                                cursor: 'pointer',
                                fontWeight: knownTokenIndices.includes(index) ? 'bold' : 'normal'
                              }}
                              title="Mark as known word"
                            >
                              ✓
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          };

          return (
            <div style={{ marginTop: 12 }}>
              {renderBandTable(lowerBandWords, 'Lower', bandColors.below, 'lower')}
              {renderBandTable(middleBandWords, 'Middle', bandColors.between, 'middle')}
              {renderBandTable(upperBandWords, 'Upper', bandColors.above, 'upper')}
            </div>
          );
        })()}
      </div>

      {/* Word Attention Heatmap toggle and display moved to bottom */}
      <div style={{marginTop: 32, marginBottom: 16}}>
        <button onClick={() => setShowWordHeatmap(v => !v)} style={{fontSize: 14, padding: '4px 12px'}}>
          {showWordHeatmap ? 'Hide' : 'Show'} Word Attention Heatmap
        </button>
        {showWordHeatmap && (
          <div style={{marginTop: 16}}>
            <b>Word Attention Heatmap (red = low, white = high):</b>
            <table style={{ borderCollapse: 'collapse', marginTop: 8 }}>
              <thead>
                <tr>
                  <th></th>
                  {displayWords.map((w, i) => (
                    <th key={i} style={{ writingMode: 'vertical-rl', fontSize: '0.9em', padding: '2px 4px' }}>{w}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {displayAttention.map((row, i) => (
                  <tr key={i}>
                    <th style={{ textAlign: 'right', fontSize: '0.9em', padding: '2px 4px' }}>{displayWords[i]}</th>
                    {row.map((val, j) => (
                      <td key={j} style={{ background: getRedWhiteColor(val, attMin, attMax), width: 24, height: 24, textAlign: 'center', fontSize: '0.8em' }}>
                        {val.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <div style={{ fontSize: '0.8em', marginTop: 4 }}>
              Each cell shows how much the column word attends to the row word.
            </div>
          </div>
        )}
      </div>

      {/* Attention Score Formulas Toggle moved to bottom */}
      <div style={{ textAlign: 'center', margin: '24px 0 8px 0' }}>
        <button onClick={() => setShowFormulas(v => !v)} style={{marginBottom: 8}}>
          {showFormulas ? 'Hide' : 'Show'} Attention Score Formulas
        </button>
      </div>
      {showFormulas && (
        <div style={{margin: '0 auto 24px auto', width: '100%'}}>
          <table style={{
            margin: '0 auto',
            borderCollapse: 'collapse',
            fontSize: '1em',
            width: '100%',
            background: '#f8f8f8',
            borderRadius: 8,
            boxSizing: 'border-box',
            padding: 0
          }}>
            <thead>
              <tr>
                <th style={{ textAlign: 'left', padding: '6px 12px', width: '18%' }}>Score</th>
                <th style={{ textAlign: 'left', padding: '6px 12px', width: '32%' }}>Formula</th>
                <th style={{ textAlign: 'left', padding: '6px 12px' }}>Explanation</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}><b>Total Provided</b> (for word <i>j</i>)</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>total_provided_j = Σ(k=1 to N) A_jk</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>Sum of row <i>j</i> in the heatmap. How much attention word <b>j</b> gives to all words (including itself).</td>
              </tr>
              <tr>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}><b>Total Received</b> (for word <i>i</i>)</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>total_received_i = Σ(k=1 to N) A_ki</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>Sum of column <i>i</i> in the heatmap. How much attention word <b>i</b> receives from all words.</td>
              </tr>
              <tr>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}><b>Total Left</b> (for word <i>i</i>)</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>total_left_i = Σ(j=1 to i-1) A_ij</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>Attention word <b>i</b> gives to words to its left (lower column indices).</td>
              </tr>
              <tr>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}><b>Total Right</b> (for word <i>i</i>)</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>total_right_i = Σ(j=i+1 to N) A_ij</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>Attention word <b>i</b> gives to words to its right (higher column indices).</td>
              </tr>
              <tr>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}><b>Heatmap</b> <i>A<sub>ij</sub></i></td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>A_ij = Σ(l=1 to L) Σ(h=1 to H) Attention^(l,h)_ij</td>
                <td style={{ textAlign: 'left', verticalAlign: 'top', padding: '6px 12px' }}>Sum of attention from word <b>j</b> to word <b>i</b> over all layers <i>L</i> and heads <i>H</i> (matrix cell in the heatmap).</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default AttentionHeatmap;
