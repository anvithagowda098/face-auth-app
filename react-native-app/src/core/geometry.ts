/**
 * geometry.ts — 2-D similarity-transform estimation for face alignment.
 *
 * ArcFace alignment warps the five detected keypoints onto a fixed template
 * using a similarity transform (uniform scale + rotation + translation, NO
 * shear, NO reflection). InsightFace does this with skimage's
 * SimilarityTransform (Umeyama). For the no-reflection similarity case there is
 * an exact closed-form least-squares solution via complex numbers — no SVD —
 * which we use here so the whole thing is a few lines of testable arithmetic.
 *
 * Given centred source points (xᵢ,yᵢ) and target points (uᵢ,vᵢ), the optimal
 * rotation+scale matrix has the form [[a,-b],[b,a]] with
 *     a = Σ(xᵢuᵢ + yᵢvᵢ) / Σ(xᵢ² + yᵢ²)
 *     b = Σ(xᵢvᵢ − yᵢuᵢ) / Σ(xᵢ² + yᵢ²)
 * and translation t = μ_dst − M·μ_src.
 */

import type { Point } from './types';

/** A 2x3 affine matrix [[a, c, tx], [b, d, ty]] mapping src -> dst. */
export type Affine = [number, number, number, number, number, number];
//                     a       b       c       d       tx      ty
// Applied as: x' = a*x + c*y + tx ;  y' = b*x + d*y + ty

function centroid(pts: Point[]): Point {
  'worklet'
  let sx = 0;
  let sy = 0;
  for (const p of pts) {
    sx += p.x;
    sy += p.y;
  }
  return { x: sx / pts.length, y: sy / pts.length };
}

/**
 * Estimate the similarity transform mapping `src` onto `dst` (least squares).
 * `src` and `dst` must be the same length and >= 2 points.
 */
export function estimateSimilarity(src: Point[], dst: Point[]): Affine {
  'worklet'
  if (src.length !== dst.length || src.length < 2) {
    throw new Error('estimateSimilarity: need matching point sets of length >= 2');
  }
  const muSrc = centroid(src);
  const muDst = centroid(dst);

  let num1 = 0; // Σ(x·u + y·v)
  let num2 = 0; // Σ(x·v − y·u)
  let den = 0; // Σ(x² + y²)
  for (let i = 0; i < src.length; i++) {
    const x = src[i].x - muSrc.x;
    const y = src[i].y - muSrc.y;
    const u = dst[i].x - muDst.x;
    const v = dst[i].y - muDst.y;
    num1 += x * u + y * v;
    num2 += x * v - y * u;
    den += x * x + y * y;
  }
  if (den === 0) {
    throw new Error('estimateSimilarity: degenerate source points (zero variance)');
  }

  const a = num1 / den; // = scale·cosθ
  const b = num2 / den; // = scale·sinθ
  // M = [[a, -b], [b, a]]
  const tx = muDst.x - (a * muSrc.x - b * muSrc.y);
  const ty = muDst.y - (b * muSrc.x + a * muSrc.y);

  return [a, b, -b, a, tx, ty];
}

/** Apply a 2x3 affine to a point. */
export function applyAffine(m: Affine, p: Point): Point {
  return {
    x: m[0] * p.x + m[2] * p.y + m[4],
    y: m[1] * p.x + m[3] * p.y + m[5],
  };
}

/** Uniform scale factor of a similarity affine (sqrt(det) of the 2x2 block). */
export function affineScale(m: Affine): number {
  return Math.sqrt(Math.abs(m[0] * m[3] - m[1] * m[2]));
}
