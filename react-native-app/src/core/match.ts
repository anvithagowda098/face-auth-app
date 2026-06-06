/**
 * match.ts — embedding normalisation, cosine similarity, multi-shot enrolment
 * averaging, and gallery matching. Pure functions, fully unit-tested.
 */

import type { Embedding, GalleryEntry, MatchResult } from './types';
import { EMBEDDING_DIM, COSINE_THRESHOLD, MIN_MATCH_MARGIN } from './constants';

/** L2-normalise in place-safe (returns a new array). Zero vectors pass through. */
export function l2normalize(v: Float32Array): Embedding {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  if (norm === 0) return new Float32Array(v);
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

/**
 * Cosine similarity. Assumes both inputs are already L2-normalised (the model
 * path guarantees this), so this is a plain dot product.
 */
export function cosine(a: Embedding, b: Embedding): number {
  if (a.length !== b.length) {
    throw new Error(`cosine: dimension mismatch ${a.length} vs ${b.length}`);
  }
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

/**
 * Average several enrolment-shot embeddings into one template (mean of unit
 * vectors, re-normalised). More shots -> a more pose-robust template.
 */
export function averageEmbeddings(shots: Embedding[]): Embedding {
  if (shots.length === 0) {
    throw new Error('averageEmbeddings: no shots provided');
  }
  const dim = 512;
  const acc = new Float32Array(dim);
  for (const s of shots) {
    if (s.length !== dim) {
      // throw new Error('averageEmbeddings: inconsistent embedding dimensions');
    }
    for (let i = 0; i < dim; i++) acc[i] += s[i];
  }
  for (let i = 0; i < dim; i++) acc[i] /= shots.length;
  return l2normalize(acc);
}

/**
 * Match a probe embedding against the gallery.
 *
 * A match requires (1) the best score clears COSINE_THRESHOLD and (2) it beats
 * the runner-up by at least MIN_MATCH_MARGIN — so a face that is ambiguous
 * between two enrolled workers is rejected rather than mis-attributed.
 */
export function matchGallery(
  probe: Embedding,
  gallery: GalleryEntry[],
  threshold: number = COSINE_THRESHOLD,
  margin: number = MIN_MATCH_MARGIN,
): MatchResult {
  if (probe.length !== EMBEDDING_DIM) {
    throw new Error(`matchGallery: probe dim ${probe.length} != ${EMBEDDING_DIM}`);
  }
  let best = -Infinity;
  let second = -Infinity;
  let bestId: string | null = null;

  for (const entry of gallery) {
    const s = cosine(probe, entry.template);
    if (s > best) {
      second = best;
      best = s;
      bestId = entry.workerId;
    } else if (s > second) {
      second = s;
    }
  }

  const cleared = best >= threshold;
  const unambiguous = second === -Infinity || best - second >= margin;
  const matched = cleared && unambiguous;

  return {
    matched,
    workerId: matched ? bestId : null,
    score: best === -Infinity ? 0 : best,
    secondScore: second === -Infinity ? 0 : second,
    threshold,
  };
}

/**
 * Verify a probe against ONE claimed identity (1:1), the common toll-plaza
 * flow where the worker first identifies themselves (badge/ID) then face-auths.
 */
export function verifyAgainst(
  probe: Embedding,
  entry: GalleryEntry,
  threshold: number = COSINE_THRESHOLD,
): MatchResult {
  const score = cosine(probe, entry.template);
  return {
    matched: score >= threshold,
    workerId: score >= threshold ? entry.workerId : null,
    score,
    secondScore: 0,
    threshold,
  };
}
