import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import React from 'react';
import { RootStackParamList } from './types'; // Tanımladığımız tipleri import ediyoruz

// Ekran bileşenlerini import ediyoruz (şimdilik boş dosyalar olabilir)
import HomeScreen from '../screens/HomeScreen/HomeScreen';
import ReportsScreen from '../screens/ReportsScreen/ReportsScreen';
import RoomDetailScreen from '../screens/RoomDetailScreen/RoomDetailScreen';
import UploadScreen from '../screens/UploadScreen/UploadScreen';

const Stack = createNativeStackNavigator<RootStackParamList>();

const AppNavigator = () => {
  return (
    <NavigationContainer>
      <Stack.Navigator initialRouteName="Home">
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{ title: 'myRoom', headerLargeTitle: true }} // Başlık ve stil
        />
        <Stack.Screen
          name="Upload"
          component={UploadScreen}
          options={{ title: 'Yeni Tarama Yükle' }}
        />
        <Stack.Screen
          name="RoomDetail"
          component={RoomDetailScreen}
          options={({ route }) => ({ title: `Oda Detayı: ${route.params.roomId}` })} // Dinamik başlık
        />
        <Stack.Screen
          name="Reports"
          component={ReportsScreen}
          options={{ title: 'Raporlar' }}
        />
        {/* Diğer ekranlar buraya eklenecek */}
      </Stack.Navigator>
    </NavigationContainer>
  );
};

export default AppNavigator;