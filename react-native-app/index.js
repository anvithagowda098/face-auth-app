/**
 * Entry point. The crypto polyfill must be imported before anything that uses
 * crypto.getRandomValues (uuid, hmac key generation, device id).
 */
import 'react-native-get-random-values';
import { AppRegistry } from 'react-native';
import App from './App';
import { name as appName } from './app.json';

AppRegistry.registerComponent(appName, () => App);
