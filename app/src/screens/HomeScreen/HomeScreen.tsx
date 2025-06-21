import { useNavigation } from '@react-navigation/native';
import { NativeStackNavigationProp } from '@react-navigation/native-stack';
import React from 'react';
import { Image, ScrollView, Text, TouchableOpacity, View } from 'react-native';
import { Colors } from '../../constants/Colors'; // Path'i kendi yapına göre ayarla
import { RootStackParamList } from '../../navigation/types'; // Path'i kendi yapına göre ayarla
import { homeScreenStyles } from './HomeScreen.styles';

type HomeScreenNavigationProp = NativeStackNavigationProp<RootStackParamList, 'Home'>;

const HomeScreen: React.FC = () => {
  const navigation = useNavigation<HomeScreenNavigationProp>();

  const handleUploadNewScan = () => {
    navigation.navigate('Upload');
  };

  const handleViewRooms = () => {
    // Burada mevcut odaların listeleneceği bir ekrana yönlendirme yapılabilir.
    // Şimdilik örnek olarak RoomDetail'e yönlendiriyoruz.
    navigation.navigate('RoomDetail', { roomId: 'Oda_Kodu_123' });
  };

  const handleViewReports = () => {
    navigation.navigate('Reports');
  };

  return (
    <ScrollView style={homeScreenStyles.scrollViewContainer} contentContainerStyle={homeScreenStyles.contentContainer}>
      <View style={homeScreenStyles.header}>
        {/* Uygulama logosu veya ikon alanı */}
        <Image
          source={require('../../assets/icons/Gemini_Generated_Image_ks6g2nks6g2nks6g-Photoroom.png')} // Kendi logonun path'ini ayarla
          style={homeScreenStyles.logo}
        />
        <Text style={homeScreenStyles.appTitle}>myRoom</Text>
      </View>

      <View style={homeScreenStyles.heroSection}>
        <Text style={homeScreenStyles.heroTitle}>Güvenli Ortamlar Yaratın</Text>
        <Text style={homeScreenStyles.heroSubtitle}>
          İç mekanlarınızdaki potansiyel güvenlik risklerini kolayca tespit edin ve yönetin.
        </Text>
      </View>

      <View style={homeScreenStyles.buttonContainer}>
        <TouchableOpacity
          style={[homeScreenStyles.actionButton, { backgroundColor: Colors.primary }]}
          onPress={handleUploadNewScan}
        >
          <Text style={homeScreenStyles.buttonText}>Yeni Tarama Başlat</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[homeScreenStyles.actionButton, { backgroundColor: Colors.secondary }]}
          onPress={handleViewRooms}
        >
          <Text style={homeScreenStyles.buttonText}>Odalarımı Görüntüle</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={[homeScreenStyles.actionButton, { backgroundColor: Colors.darkGray }]}
          onPress={handleViewReports}
        >
          <Text style={homeScreenStyles.buttonText}>Raporları İncele</Text>
        </TouchableOpacity>
      </View>

      {/* Ek Bilgi veya Quick Stats alanı eklenebilir */}
      <View style={homeScreenStyles.infoSection}>
        <Text style={homeScreenStyles.infoTitle}>Neden myRoom?</Text>
        <Text style={homeScreenStyles.infoText}>
          myRoom, LiDAR taramalarınızı ve fotoğraflarınızı analiz ederek, OSHA standartlarına uygun olarak otomatik tehlike tespiti yapar. Düşme risklerinden elektrik güvenliğine kadar her şeyi kontrol edin ve kapsamlı güvenlik skorları alın.
        </Text>
      </View>

    </ScrollView>
  );
};

export default HomeScreen;