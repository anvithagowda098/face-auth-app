/**
 * LivenessGuide.tsx — the challenge strip shown during liveness: the active
 * prompt plus a row of step chips that tick off as each challenge is satisfied.
 */

import { View, StyleSheet } from 'react-native';
import { Text, Icon } from '../ui';
import { palette, spacing, radius } from '../theme';
import { CHALLENGE_PROMPT, type ChallengeId } from '../engine/LivenessEngine';

interface Props {
  index: number; // number completed
}

export default function LivenessGuide({ index }: Props) {
  return (
    <View style={styles.wrap}>
      <View style={styles.row}>
        {Array.from(Object.keys(CHALLENGE_PROMPT)).map((c, i) => {
          const done = i < index;
          const active = i === index;
          return (
            <View
              key={c}
              style={[
                styles.chip,
                done && styles.chipDone,
                active && styles.chipActive,
              ]}
            >
              {done ? (
                <Icon name="check" size={14} color={palette.textInverse} strokeWidth={2.5} />
              ) : (
                <Text variant="caption" color={active ? palette.accent : palette.textMuted}>
                  {i + 1}
                </Text>
              )}
              <Text
                variant="label"
                color={done ? palette.textInverse : active ? palette.text : palette.textMuted}
              >
                {CHALLENGE_PROMPT[c as ChallengeId]}
              </Text>
            </View>
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrap: { alignItems: 'center' },
  row: { flexDirection: 'row', flexWrap: 'wrap', gap: spacing.sm, justifyContent: 'center' },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.xs + 2,
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: radius.pill,
    backgroundColor: palette.scrim,
    borderWidth: 1,
    borderColor: palette.border,
  },
  chipActive: { borderColor: palette.accentBorder, backgroundColor: palette.accentSoft },
  chipDone: { backgroundColor: palette.accent, borderColor: palette.accent },
});
