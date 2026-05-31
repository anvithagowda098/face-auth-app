/**
 * Design tokens — a single source of truth for the visual language.
 *
 * Direction: "security-ops dark", done properly. A muted slate palette (not
 * pure black, not neon), one restrained accent, a typographic scale with tight
 * tracking on headings, a 4-pt spacing grid, and soft elevation. No emoji, no
 * decorative gradients. Everything reads as instrument-panel, not toy.
 */

export const palette = {
  // Backgrounds — layered slate, subtle separation between planes.
  bg: '#0B0F14', // app base
  surface: '#111824', // cards / sheets
  surfaceAlt: '#161F2E', // raised rows, inputs
  surfaceHi: '#1C2738', // hover / pressed / selected

  // Hairlines.
  border: '#22304A',
  borderStrong: '#2E3F5C',

  // Text.
  text: '#E7EDF5', // primary
  textSecondary: '#9AA7BD', // secondary
  textMuted: '#5E6B82', // captions, disabled
  textInverse: '#08111B', // on accent fills

  // Accent — a controlled cyan-teal. Trust + technical, not lime.
  accent: '#2DD4BF',
  accentDim: '#1C8C81',
  accentSoft: 'rgba(45,212,191,0.12)', // tinted fills
  accentBorder: 'rgba(45,212,191,0.32)',

  // Semantic.
  success: '#34D399',
  successSoft: 'rgba(52,211,153,0.12)',
  danger: '#F26D6D',
  dangerSoft: 'rgba(242,109,109,0.12)',
  warning: '#F2B441',
  warningSoft: 'rgba(242,180,65,0.12)',
  info: '#5BA8F5',

  // Overlay scrims (over the live camera).
  scrim: 'rgba(7,11,18,0.66)',
  scrimSoft: 'rgba(7,11,18,0.38)',
} as const;

export const spacing = {
  xs: 4,
  sm: 8,
  md: 12,
  lg: 16,
  xl: 24,
  xxl: 32,
  xxxl: 48,
} as const;

export const radius = {
  sm: 8,
  md: 12,
  lg: 16,
  xl: 22,
  pill: 999,
} as const;

/**
 * Type scale. `weight` and `tracking` are tuned per role: headings are tight,
 * numeric/mono values use tabular-ish spacing via fontVariant where supported.
 */
export const type = {
  display: { fontSize: 30, lineHeight: 36, fontWeight: '700' as const, letterSpacing: -0.5 },
  h1: { fontSize: 22, lineHeight: 28, fontWeight: '700' as const, letterSpacing: -0.3 },
  h2: { fontSize: 17, lineHeight: 23, fontWeight: '600' as const, letterSpacing: -0.2 },
  body: { fontSize: 15, lineHeight: 21, fontWeight: '400' as const, letterSpacing: 0 },
  bodyStrong: { fontSize: 15, lineHeight: 21, fontWeight: '600' as const, letterSpacing: 0 },
  label: { fontSize: 13, lineHeight: 17, fontWeight: '500' as const, letterSpacing: 0.1 },
  caption: { fontSize: 11, lineHeight: 15, fontWeight: '500' as const, letterSpacing: 0.3 },
  // Numeric "instrument" readout (stats, scores, latency).
  mono: { fontSize: 26, lineHeight: 30, fontWeight: '700' as const, letterSpacing: -0.4 },
  monoSm: { fontSize: 15, lineHeight: 19, fontWeight: '600' as const, letterSpacing: 0 },
  overline: { fontSize: 10, lineHeight: 13, fontWeight: '700' as const, letterSpacing: 1.4 },
} as const;

export const elevation = {
  card: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.28,
    shadowRadius: 16,
    elevation: 8,
  },
  sheet: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -8 },
    shadowOpacity: 0.4,
    shadowRadius: 24,
    elevation: 24,
  },
} as const;

export type TypeRole = keyof typeof type;
