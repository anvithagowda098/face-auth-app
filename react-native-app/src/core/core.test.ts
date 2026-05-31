/**
 * Pure-logic unit tests for the face-recognition core.
 * Run with:  npx tsx src/core/core.test.ts
 *
 * These cover everything except the live ORT session (embedder.ts), which
 * needs the native runtime. No react-native imports are pulled in here.
 */

import assert from 'node:assert/strict';

import { estimateSimilarity, applyAffine, affineScale, type Affine } from './geometry';
import { ARCFACE_TEMPLATE, orderLandmarks, alignmentTransform } from './align';
import { toNCHW } from './preprocess';
import {
  l2normalize,
  cosine,
  averageEmbeddings,
  matchGallery,
  verifyAgainst,
} from './match';
import { INPUT_SIZE, EMBEDDING_DIM, PIXEL_MEAN, PIXEL_STD, COSINE_THRESHOLD } from './constants';
import type { FaceLandmarks, GalleryEntry, Point } from './types';

let passed = 0;
function test(name: string, fn: () => void) {
  try {
    fn();
    passed++;
    console.log(`  ok  ${name}`);
  } catch (e) {
    console.error(`FAIL  ${name}\n      ${(e as Error).message}`);
    process.exitCode = 1;
  }
}
const approx = (a: number, b: number, eps = 1e-6) =>
  assert.ok(Math.abs(a - b) <= eps, `expected ${a} ≈ ${b}`);

function rand512(seed: number): Float32Array {
  // deterministic pseudo-random (no Math.random) so tests are reproducible
  const v = new Float32Array(EMBEDDING_DIM);
  let s = seed >>> 0;
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    v[i] = (s / 0xffffffff) * 2 - 1;
  }
  return l2normalize(v);
}

// ── geometry ────────────────────────────────────────────────────────────────
test('estimateSimilarity recovers identity', () => {
  const pts: Point[] = [
    { x: 0, y: 0 },
    { x: 1, y: 0 },
    { x: 0, y: 1 },
  ];
  const m = estimateSimilarity(pts, pts);
  approx(m[0], 1);
  approx(m[1], 0);
  approx(m[4], 0, 1e-9);
  approx(m[5], 0, 1e-9);
});

test('estimateSimilarity recovers scale+rotation+translation', () => {
  const src: Point[] = [
    { x: 10, y: 5 },
    { x: 30, y: 5 },
    { x: 20, y: 25 },
    { x: 12, y: 40 },
    { x: 28, y: 40 },
  ];
  const theta = Math.PI / 6; // 30°
  const scale = 1.7;
  const tx = 13;
  const ty = -7;
  const known: Affine = [
    scale * Math.cos(theta),
    scale * Math.sin(theta),
    -scale * Math.sin(theta),
    scale * Math.cos(theta),
    tx,
    ty,
  ];
  const dst = src.map(p => applyAffine(known, p));
  const est = estimateSimilarity(src, dst);
  for (let i = 0; i < 6; i++) approx(est[i], known[i], 1e-6);
  approx(affineScale(est), scale, 1e-6);
});

// ── align ──────────────────────────────────────────────────────────────────
test('aligning the template onto itself is ~identity', () => {
  const lm: FaceLandmarks = {
    leftEye: ARCFACE_TEMPLATE[0],
    rightEye: ARCFACE_TEMPLATE[1],
    nose: ARCFACE_TEMPLATE[2],
    leftMouth: ARCFACE_TEMPLATE[3],
    rightMouth: ARCFACE_TEMPLATE[4],
  };
  const m = alignmentTransform(lm);
  approx(affineScale(m), 1, 1e-6);
  for (const p of ARCFACE_TEMPLATE) {
    const q = applyAffine(m, p);
    approx(q.x, p.x, 1e-4);
    approx(q.y, p.y, 1e-4);
  }
});

test('orderLandmarks fixes swapped left/right (mirrored camera)', () => {
  // eyes/mouth provided with subject-relative (swapped) labels
  const lm: FaceLandmarks = {
    leftEye: { x: 73, y: 51 }, // actually image-right
    rightEye: { x: 38, y: 51 }, // actually image-left
    nose: { x: 56, y: 71 },
    leftMouth: { x: 70, y: 92 },
    rightMouth: { x: 41, y: 92 },
  };
  const ordered = orderLandmarks(lm);
  assert.ok(ordered[0].x < ordered[1].x, 'eye[0] must be image-left');
  assert.ok(ordered[3].x < ordered[4].x, 'mouth[0] must be image-left');
});

test('alignment warps a real-ish detection close to template', () => {
  // a face shifted+scaled+slightly rotated in the frame
  const lm: FaceLandmarks = {
    leftEye: { x: 220, y: 300 },
    rightEye: { x: 320, y: 308 },
    nose: { x: 268, y: 360 },
    leftMouth: { x: 232, y: 420 },
    rightMouth: { x: 312, y: 426 },
  };
  const m = alignmentTransform(lm);
  const src = orderLandmarks(lm);
  let maxErr = 0;
  for (let i = 0; i < 5; i++) {
    const q = applyAffine(m, src[i]);
    maxErr = Math.max(maxErr, Math.hypot(q.x - ARCFACE_TEMPLATE[i].x, q.y - ARCFACE_TEMPLATE[i].y));
  }
  // least-squares residual onto a frontal template should be small (px)
  assert.ok(maxErr < 6, `alignment residual too large: ${maxErr.toFixed(2)}px`);
});

// ── preprocess ──────────────────────────────────────────────────────────────
test('toNCHW de-interleaves and normalises', () => {
  const n = INPUT_SIZE * INPUT_SIZE;
  const rgb = new Uint8Array(n * 3);
  // pixel 0 = (255,0,127.5→127), rest 127.5-ish
  rgb[0] = 255;
  rgb[1] = 0;
  rgb[2] = 128;
  const t = toNCHW(rgb);
  assert.equal(t.length, n * 3);
  approx(t[0], (255 - PIXEL_MEAN) / PIXEL_STD); // R plane, px0
  approx(t[n], (0 - PIXEL_MEAN) / PIXEL_STD); // G plane, px0
  approx(t[2 * n], (128 - PIXEL_MEAN) / PIXEL_STD); // B plane, px0
});

test('toNCHW rejects wrong-sized buffers', () => {
  assert.throws(() => toNCHW(new Uint8Array(10)));
});

// ── match ───────────────────────────────────────────────────────────────────
test('l2normalize yields unit length', () => {
  const v = l2normalize(new Float32Array([3, 4]));
  approx(Math.hypot(v[0], v[1]), 1);
});

test('cosine: identical=1, orthogonal=0, opposite=-1', () => {
  const a = l2normalize(new Float32Array([1, 2, 3, 4]));
  approx(cosine(a, a), 1, 1e-6);
  approx(cosine(l2normalize(new Float32Array([1, 0])), l2normalize(new Float32Array([0, 1]))), 0, 1e-9);
  const b = l2normalize(new Float32Array([1, 1]));
  approx(cosine(b, l2normalize(new Float32Array([-1, -1]))), -1, 1e-6);
});

test('averageEmbeddings of one vector is itself', () => {
  const e = rand512(7);
  const avg = averageEmbeddings([e]);
  approx(cosine(e, avg), 1, 1e-6);
});

test('averaged template is closer to its shots than to an impostor', () => {
  const base = rand512(11);
  // jittered shots of the same identity
  const shots = [0, 1, 2, 3, 4].map(k => {
    const j = new Float32Array(base);
    const noise = rand512(100 + k);
    for (let i = 0; i < j.length; i++) j[i] = base[i] * 0.9 + noise[i] * 0.1;
    return l2normalize(j);
  });
  const template = averageEmbeddings(shots);
  const impostor = rand512(999);
  assert.ok(cosine(template, shots[0]) > cosine(template, impostor));
});

test('matchGallery picks the right identity and respects threshold', () => {
  const a = rand512(1);
  const b = rand512(2);
  const gallery: GalleryEntry[] = [
    { workerId: 'EMP_001', template: a, enrolledAt: '', shots: 5 },
    { workerId: 'EMP_002', template: b, enrolledAt: '', shots: 5 },
  ];
  const hit = matchGallery(a, gallery);
  assert.equal(hit.matched, true);
  assert.equal(hit.workerId, 'EMP_001');
  approx(hit.score, 1, 1e-6);

  const stranger = rand512(424242);
  const miss = matchGallery(stranger, gallery);
  assert.ok(miss.score < COSINE_THRESHOLD, `stranger score ${miss.score} should be sub-threshold`);
  assert.equal(miss.matched, false);
  assert.equal(miss.workerId, null);
});

test('matchGallery rejects ambiguous probe (margin gate)', () => {
  const a = rand512(1);
  // build b very close to a so the probe is ~equidistant
  const b = new Float32Array(a);
  const noise = rand512(5);
  for (let i = 0; i < b.length; i++) b[i] = a[i] * 0.999 + noise[i] * 0.001;
  const bn = l2normalize(b);
  const gallery: GalleryEntry[] = [
    { workerId: 'EMP_001', template: a, enrolledAt: '', shots: 5 },
    { workerId: 'EMP_002', template: bn, enrolledAt: '', shots: 5 },
  ];
  const res = matchGallery(a, gallery, 0.2, 0.05);
  // a matches EMP_001 strongly but EMP_002 is nearly as close -> ambiguous
  assert.equal(res.matched, false, 'should reject when margin is tiny');
});

test('verifyAgainst is a clean 1:1 decision', () => {
  const a = rand512(3);
  const entry: GalleryEntry = { workerId: 'EMP_007', template: a, enrolledAt: '', shots: 5 };
  assert.equal(verifyAgainst(a, entry).matched, true);
  assert.equal(verifyAgainst(rand512(8), entry).matched, false);
});

console.log(`\n${passed} checks passed.`);
