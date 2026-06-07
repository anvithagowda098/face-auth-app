/**
 * Icon.tsx — a small, consistent set of stroke icons (24x24 grid) rendered with
 * react-native-svg. Replaces the emoji that made the old UI read as a toy.
 *
 * All icons share one geometry: 1.75 stroke, round caps/joins, currentColor via
 * the `color` prop. Add new glyphs by adding a path entry — keep the visual
 * weight uniform.
 */

import React from 'react';
import Svg, { Path, Circle, Line, Polyline } from 'react-native-svg';
import { palette } from '../theme';

export type IconName =
  | 'shield-check'
  | 'face-scan'
  | 'user-plus'
  | 'users'
  | 'bolt'
  | 'check'
  | 'x'
  | 'check-circle'
  | 'x-circle'
  | 'arrow-left'
  | 'chevron-right'
  | 'cloud-sync'
  | 'wifi-off'
  | 'lock'
  | 'refresh'
  | 'alert'
  | 'cpu'
  | 'activity';

interface IconProps {
  name: IconName;
  size?: number;
  color?: string;
  strokeWidth?: number;
}

export default function Icon({
  name,
  size = 22,
  color = palette.text,
  strokeWidth = 1.75,
}: IconProps) {
  const common = {
    stroke: color,
    strokeWidth,
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    fill: 'none',
  };
  return (
    <Svg width={size} height={size} viewBox="0 0 24 24">
      {render(name, common)}
    </Svg>
  );
}

function render(name: IconName, p: object) {
  switch (name) {
    case 'shield-check':
      return (
        <>
          <Path {...p} d="M12 3l7 3v5c0 4.4-3 7.6-7 9-4-1.4-7-4.6-7-9V6l7-3z" />
          <Polyline {...p} points="9 12 11.2 14.2 15.2 9.6" />
        </>
      );
    case 'face-scan':
      return (
        <>
          <Path {...p} d="M4 8V6a2 2 0 012-2h2M16 4h2a2 2 0 012 2v2M20 16v2a2 2 0 01-2 2h-2M8 20H6a2 2 0 01-2-2v-2" />
          <Circle {...p} cx="9.5" cy="11" r="0.6" fill={(p as any).stroke} />
          <Circle {...p} cx="14.5" cy="11" r="0.6" fill={(p as any).stroke} />
          <Path {...p} d="M9.5 15c1.5 1.2 3.5 1.2 5 0" />
        </>
      );
    case 'user-plus':
      return (
        <>
          <Circle {...p} cx="10" cy="8" r="3.2" />
          <Path {...p} d="M4 20c0-3.3 2.7-6 6-6 1.2 0 2.3.35 3.2.95" />
          <Line {...p} x1="18" y1="14" x2="18" y2="20" />
          <Line {...p} x1="15" y1="17" x2="21" y2="17" />
        </>
      );
    case 'users':
      return (
        <>
          <Circle {...p} cx="9" cy="8" r="3" />
          <Path {...p} d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6" />
          <Path {...p} d="M16 5.2A3 3 0 0118 11M21 20c0-2.6-1.5-4.8-3.6-5.7" />
        </>
      );
    case 'bolt':
      return <Path {...p} d="M13 3L5 13h5l-1 8 8-10h-5l1-8z" />;
    case 'check':
      return <Polyline {...p} points="5 12.5 10 17.5 19 7" />;
    case 'x':
      return (
        <>
          <Line {...p} x1="6" y1="6" x2="18" y2="18" />
          <Line {...p} x1="18" y1="6" x2="6" y2="18" />
        </>
      );
    case 'check-circle':
      return (
        <>
          <Circle {...p} cx="12" cy="12" r="9" />
          <Polyline {...p} points="8 12 11 15 16 9" />
        </>
      );
    case 'x-circle':
      return (
        <>
          <Circle {...p} cx="12" cy="12" r="9" />
          <Line {...p} x1="9" y1="9" x2="15" y2="15" />
          <Line {...p} x1="15" y1="9" x2="9" y2="15" />
        </>
      );
    case 'arrow-left':
      return (
        <>
          <Line {...p} x1="19" y1="12" x2="5" y2="12" />
          <Polyline {...p} points="11 6 5 12 11 18" />
        </>
      );
    case 'chevron-right':
      return <Polyline {...p} points="9 5 16 12 9 19" />;
    case 'cloud-sync':
      return (
        <>
          <Path {...p} d="M7 18a4 4 0 01-.5-7.97A5 5 0 0116.9 9.2 3.5 3.5 0 0117 18" />
          <Polyline {...p} points="10 14 12 12 14 14" />
          <Line {...p} x1="12" y1="12" x2="12" y2="17" />
        </>
      );
    case 'wifi-off':
      return (
        <>
          <Path {...p} d="M3 8.5a14 14 0 014.5-2.8M10 5.2a14 14 0 0111 3.3M6 12a9 9 0 013-1.9M14.5 10.4A9 9 0 0118 12M9 15.5a4.5 4.5 0 015.2-.7" />
          <Line {...p} x1="12" y1="19" x2="12" y2="19" />
          <Line {...p} x1="4" y1="4" x2="20" y2="20" />
        </>
      );
    case 'lock':
      return (
        <>
          <Path {...p} d="M6 11h12v8H6z" />
          <Path {...p} d="M8.5 11V8a3.5 3.5 0 017 0v3" />
        </>
      );
    case 'refresh':
      return (
        <>
          <Path {...p} d="M20 11a8 8 0 00-14-4.5L4 8" />
          <Polyline {...p} points="4 4 4 8 8 8" />
          <Path {...p} d="M4 13a8 8 0 0014 4.5L20 16" />
          <Polyline {...p} points="20 20 20 16 16 16" />
        </>
      );
    case 'alert':
      return (
        <>
          <Path {...p} d="M12 4l9 16H3l9-16z" />
          <Line {...p} x1="12" y1="10" x2="12" y2="14" />
          <Line {...p} x1="12" y1="17" x2="12" y2="17" />
        </>
      );
    case 'cpu':
      return (
        <>
          <Path {...p} d="M7 7h10v10H7z" />
          <Path {...p} d="M10 10h4v4h-4z" />
          <Path {...p} d="M9 4v2M12 4v2M15 4v2M9 18v2M12 18v2M15 18v2M4 9h2M4 12h2M4 15h2M18 9h2M18 12h2M18 15h2" />
        </>
      );
    case 'activity':
      return <Polyline {...p} points="3 12 8 12 11 5 14 19 17 12 21 12" />;
    default:
      return null;
  }
}
