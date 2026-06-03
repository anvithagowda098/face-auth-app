/**
 * navigation.tsx — a ~50-line stack router that replaces @react-navigation
 * (native + native-stack + screens + safe-area-context) for this 3-screen app.
 *
 * It exposes the small slice of the React Navigation API the screens actually
 * use: navigate(name, params), goBack(), addListener('focus', cb), route.params.
 * Swap it back for @react-navigation later by restoring the prop types — the
 * screen call-sites are unchanged.
 */
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';

/** Route params shared across screens. */
export type RootStackParamList = {
  Home: undefined;
  Verify: { workerId?: string };
  Enrol: undefined;
};

export type RouteName = keyof RootStackParamList;

export interface Navigation {
  navigate: <N extends RouteName>(name: N, params?: RootStackParamList[N]) => void;
  goBack: () => void;
  /** Only 'focus' is implemented; returns an unsubscribe fn. */
  addListener: (type: 'focus', cb: () => void) => () => void;
}

export interface Route<N extends RouteName> {
  name: N;
  params: RootStackParamList[N];
}

export interface ScreenProps<N extends RouteName> {
  navigation: Navigation;
  route: Route<N>;
}

type ScreenComponent<N extends RouteName> = (props: ScreenProps<N>) => React.ReactElement | null;
export type ScreenMap = { [N in RouteName]: ScreenComponent<N> };

interface Entry {
  name: RouteName;
  params: unknown;
  key: number;
}

const NavContext = createContext<Navigation | null>(null);
export const useNavigation = (): Navigation => {
  const nav = useContext(NavContext);
  if (!nav) throw new Error('useNavigation must be used inside <Router>');
  return nav;
};

export function Router({
  screens,
  initialRouteName = 'Home' as RouteName,
}: {
  screens: ScreenMap;
  initialRouteName?: RouteName;
}) {
  const seq = useRef(1);
  const [stack, setStack] = useState<Entry[]>([
    { name: initialRouteName, params: undefined, key: 0 },
  ]);
  // focus listeners, keyed by route name
  const focusListeners = useRef<Map<RouteName, Set<() => void>>>(new Map());

  const top = stack[stack.length - 1]!;
  // Track the top route name so addListener (called from the mounted top screen)
  // registers against the right screen.
  const topNameRef = useRef<RouteName>(top.name);
  topNameRef.current = top.name;

  const navigation = useMemo<Navigation>(
    () => ({
      navigate: (name, params) =>
        setStack(s => [...s, { name, params, key: seq.current++ }]),
      goBack: () => setStack(s => (s.length > 1 ? s.slice(0, -1) : s)),
      addListener: (type, cb) => {
        if (type !== 'focus') return () => {};
        // The screen calling this is always the current top, so register the
        // listener against the top route name.
        const name = topNameRef.current;
        const map = focusListeners.current;
        const set = map.get(name) ?? new Set();
        set.add(cb);
        map.set(name, set);
        return () => set.delete(cb);
      },
    }),
    [],
  );

  // Fire 'focus' for the top screen whenever it becomes top (mount or pop-back).
  useEffect(() => {
    const set = focusListeners.current.get(top.name);
    set?.forEach(cb => cb());
  }, [top.key, top.name]);

  const route = { name: top.name, params: top.params } as Route<RouteName>;
  const Screen = screens[top.name] as ScreenComponent<RouteName>;

  return (
    <NavContext.Provider value={navigation}>
      <Screen navigation={navigation} route={route} />
    </NavContext.Provider>
  );
}
