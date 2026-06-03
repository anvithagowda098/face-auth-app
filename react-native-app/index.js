/**
 * Entry point (Expo). registerRootComponent calls AppRegistry.registerComponent
 * and sets up the dev-client / native root for both Expo Go-less dev builds and
 * release builds.
 *
 * No crypto polyfill needed: ids/keys use expo-crypto (getRandomBytes / randomUUID).
 */
import { registerRootComponent } from 'expo';
import App from './App';

registerRootComponent(App);
