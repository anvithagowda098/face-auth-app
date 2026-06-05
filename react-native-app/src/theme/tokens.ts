export const palette = {
  // Backgrounds — cleaned up for the white and blue theme.
  bg: '#FFFFFF',           // app base remains white
  surface: '#031B4D',      // deep navy for cards
  surfaceAlt: '#052566',   // raised rows, inputs
  surfaceHi: '#082F80',    // hover / pressed / selected
 
  // Hairlines.
  border: '#1A3B7A',
  borderStrong: '#2A4B8A',
 
  // Text — updated to be light and crisp!
  text: '#FFFFFF',          // Pure white for maximum brightness on your dark cards
  textSecondary: '#D0DFF2', // Very light, soft blue for secondary text
  textMuted: '#9EBAE6',     // Light muted blue for captions and disabled states
  textInverse: '#003087',   // NHAI Navy for text that sits on light/white backgrounds 
  textInverse: '#FFFFFF',
  
  // Accent — Switched from Orange to NHAI Blue.
  accent: '#003087',        // The main navy blue
  accentDim: '#002266',     // Darker navy for hover
  accentSoft: 'rgba(0,48,135,0.12)', // Faint navy overlay
  accentBorder: 'rgba(0,48,135,0.32)',
 
  // Secondary accent — retained or used for softer highlights.
  navy: '#003087',
  navySoft: 'rgba(0,48,135,0.20)',
  navyBorder: 'rgba(0,48,135,0.40)',
 
  // Semantic.
  success: '#3366FF',       // Blue for success
  successSoft: 'rgba(51,102,255,0.12)',
  danger: '#6680B3',        // Muted blue for danger (or keep text muted)
  dangerSoft: 'rgba(102,128,179,0.12)',
  warning: '#3366FF',       // Blue for warning
  warningSoft: 'rgba(51,102,255,0.12)',
  info: '#3366FF',          // Standard blue for info
 
  // Overlay scrims (over the live camera) — shifted to blue tint.
  scrim: 'rgba(0,26,77,0.70)',
  scrimSoft: 'rgba(0,26,77,0.40)',
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
 
export const type = {
  display: { fontSize: 30, lineHeight: 36, fontWeight: '700' as const, letterSpacing: -0.5 },
  h1: { fontSize: 22, lineHeight: 28, fontWeight: '700' as const, letterSpacing: -0.3 },
  h2: { fontSize: 17, lineHeight: 23, fontWeight: '600' as const, letterSpacing: -0.2 },
  body: { fontSize: 15, lineHeight: 21, fontWeight: '400' as const, letterSpacing: 0 },
  bodyStrong: { fontSize: 15, lineHeight: 21, fontWeight: '600' as const, letterSpacing: 0 },
  label: { fontSize: 13, lineHeight: 17, fontWeight: '500' as const, letterSpacing: 0.1 },
  caption: { fontSize: 11, lineHeight: 15, fontWeight: '500' as const, letterSpacing: 0.3 },
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
