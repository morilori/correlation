// This is a mock function to generate a random attention matrix for demo purposes.
// Replace this with real attention data from your backend or LLM.
export function getMockAttention(input: string) {
  const words = input.trim().split(/\s+/);
  const n = words.length;
  // Generate a random attention matrix (rows: attended, cols: attending)
  const attention = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => {
      // Diagonal gets higher value, others random
      if (i === j) return 0.7 + 0.3 * Math.random();
      return 0.1 + 0.6 * Math.random();
    })
  );
  // Normalize columns to sum to 1 (each word's attention distribution)
  for (let j = 0; j < n; ++j) {
    let colSum = 0;
    for (let i = 0; i < n; ++i) colSum += attention[i][j];
    for (let i = 0; i < n; ++i) attention[i][j] /= colSum;
  }
  return { words, attention };
}
