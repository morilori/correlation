"""Complete analysis utilities for comprehension modeling.

This module currently provides a constant-free implementation of the
directional comprehension effort model based on an attention matrix and
optional knownness vector.
"""

from __future__ import annotations

import numpy as np


def _safe_normalize(vec: np.ndarray) -> np.ndarray:
	"""Normalize a vector to 0-1; return zeros if constant or empty."""
	if vec.size == 0:
		return vec
	vmin = np.min(vec)
	vmax = np.max(vec)
	if vmax - vmin <= 0:
		return np.zeros_like(vec)
	return (vec - vmin) / (vmax - vmin)


def comprehension_effort(attention_matrix: np.ndarray, knownness: np.ndarray | None = None):
	"""
	Computes Integration (receiving) and Contribution (providing) scores for each word
	using directional locality weighting from BERT attention.

	Parameters
	----------
	attention_matrix : np.ndarray
		Square matrix [n_words x n_words], where attention_matrix[target, source]
		is the attention the target word receives from the source word.
	knownness : np.ndarray or None
		Vector of length n_words with values in [0,1] indicating how well the word is known.
		If None, all words are assumed fully known.

	Returns
	-------
	integration : np.ndarray
		Normalized (0–1) context-based incoming meaning score per word (direction- and distance-weighted).
	contribution : np.ndarray
		Normalized (0–1) context-based outgoing meaning score per word (direction- and distance-weighted).
	effort : np.ndarray
		Combined effort score = (1 - integration) + (1 - contribution), normalized to 0–1.
	"""

	# Validate inputs
	if attention_matrix.ndim != 2 or attention_matrix.shape[0] != attention_matrix.shape[1]:
		raise ValueError("attention_matrix must be a square 2D array [n x n]")
	attention_matrix = attention_matrix.astype(float, copy=False)
	n_words = attention_matrix.shape[0]
	positions = np.arange(n_words)

	# If knownness is not provided, assume all words are fully known
	if knownness is None:
		knownness = np.ones(n_words, dtype=float)
	else:
		knownness = np.asarray(knownness, dtype=float)
		if knownness.shape != (n_words,):
			raise ValueError("knownness must be a 1D array of length n_words")

	# ----- INTEGRATION (receiving from context only) -----
	# Left integration: reward close left sources
	left_mask = positions[np.newaxis, :] < positions[:, np.newaxis]
	left_distance = np.where(left_mask, positions[:, np.newaxis] - positions[np.newaxis, :], 0)
	left_integration = np.where(
		left_mask,
		attention_matrix * knownness[np.newaxis, :] / (1 + left_distance),
		0
	).sum(axis=1)

	# Right integration: costlier, so discounted by 0.5
	right_mask = positions[np.newaxis, :] > positions[:, np.newaxis]
	right_distance = np.where(right_mask, positions[np.newaxis, :] - positions[:, np.newaxis], 0)
	right_integration = np.where(
		right_mask,
		attention_matrix * knownness[np.newaxis, :] / (1 + right_distance),
		0
	).sum(axis=1)

	integration_raw = left_integration + 0.5 * right_integration
	integration = integration_raw / integration_raw.max() if integration_raw.max() > 0 else np.zeros(n_words)

	# ----- CONTRIBUTION (providing to context) -----
	outgoing_meaning = (attention_matrix.T * knownness).T

	# Right contributions: reward close right targets
	right_mask_c = positions[np.newaxis, :] > positions[:, np.newaxis]
	right_distance_c = np.where(right_mask_c, positions[np.newaxis, :] - positions[:, np.newaxis], 0)
	right_contrib = np.where(
		right_mask_c,
		outgoing_meaning / (1 + right_distance_c),
		0
	).sum(axis=1)

	# Left contributions: discounted by 0.5
	left_mask_c = positions[np.newaxis, :] < positions[:, np.newaxis]
	left_distance_c = np.where(left_mask_c, positions[:, np.newaxis] - positions[np.newaxis, :], 0)
	left_contrib = np.where(
		left_mask_c,
		outgoing_meaning / (1 + left_distance_c),
		0
	).sum(axis=1)

	if right_contrib.max() > 0:
		right_contrib /= right_contrib.max()
	if left_contrib.max() > 0:
		left_contrib /= left_contrib.max()

	contribution_raw = right_contrib - 0.5 * left_contrib
	min_raw, max_raw = contribution_raw.min(), contribution_raw.max()
	contribution = (contribution_raw - min_raw) / (max_raw - min_raw) if max_raw > min_raw else np.zeros(n_words)

	# ----- EFFORT -----
	effort_raw = (1 - integration) + (1 - contribution)
	effort = effort_raw / effort_raw.max() if effort_raw.max() > 0 else np.zeros(n_words)

	return integration, contribution, effort


__all__ = ["comprehension_effort"]

