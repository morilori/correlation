// Given words and an attention matrix, rearrange the words so that each word has higher total attention to its left than to its right.
// Returns the new order of words (indices) and the rearranged attention matrix.
export function rearrangeWordsLeftAttention(words: string[], attention: number[][]) {
  const n = words.length;
  // Start with the original order
  let order = Array.from({ length: n }, (_, i) => i);

  // Helper: compute left/right attention for a given order
  function leftRightScore(order: number[]) {
    let score = 0;
    for (let idx = 0; idx < n; ++idx) {
      const i = order[idx];
      let left = 0, right = 0;
      for (let jdx = 0; jdx < n; ++jdx) {
        const j = order[jdx];
        if (jdx < idx) left += attention[i][j];
        if (jdx > idx) right += attention[i][j];
      }
      score += left - right;
    }
    return score;
  }

  // Greedy local search: try swapping any two words, keep swap if score improves
  let improved = true;
  while (improved) {
    improved = false;
    for (let i = 0; i < n; ++i) {
      for (let j = i + 1; j < n; ++j) {
        const newOrder = order.slice();
        [newOrder[i], newOrder[j]] = [newOrder[j], newOrder[i]];
        if (leftRightScore(newOrder) > leftRightScore(order)) {
          order = newOrder;
          improved = true;
        }
      }
    }
  }

  // Rearranged words and attention
  const newWords = order.map(i => words[i]);
  const newAttention = order.map(i => order.map(j => attention[i][j]));
  return { newWords, newAttention, order };
}
