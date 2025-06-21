import { StatusBar } from 'expo-status-bar';
import React from 'react';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import AppNavigator from './src/navigation/AppNavigator';

export default function App() {
  return (
    <SafeAreaProvider>
      <StatusBar style="dark" /> {/* Status bar stilini ayarlayabiliriz */}
      <AppNavigator /> {/* Ana navigasyonumuzu burada render ediyoruz */}
    </SafeAreaProvider>
  );
}