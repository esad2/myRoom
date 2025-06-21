import { StyleSheet } from 'react-native';
import { Colors } from '../../constants/Colors'; // Path'i kendi yapına göre ayarla
import { Fonts } from '../../constants/Fonts'; // Path'i kendi yapına göre ayarla
import { Layout } from '../../constants/Layout'; // Path'i kendi yapına göre ayarla

export const homeScreenStyles = StyleSheet.create({
  scrollViewContainer: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  contentContainer: {
    flexGrow: 1,
    alignItems: 'center',
    paddingBottom: Layout.spacing.xLarge, // ScrollView içeriği için alttan boşluk
  },
  header: {
    width: '100%',
    paddingVertical: Layout.spacing.large,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: Colors.surface, // Beyaz veya hafif gri başlık arka planı
    borderBottomWidth: 1,
    borderBottomColor: Colors.lightGray,
    paddingTop: Layout.spacing.xLarge * 1.5, // Status bar boşluğu için
  },
  logo: {
    width: 60, // Logo boyutu
    height: 60,
    resizeMode: 'contain',
    marginBottom: Layout.spacing.small,
  },
  appTitle: {
    fontSize: Fonts.size.xxLarge,
    fontWeight: Fonts.weight.bold,
    color: Colors.primary,
  },
  heroSection: {
    paddingVertical: Layout.spacing.xLarge,
    paddingHorizontal: Layout.paddingHorizontal,
    alignItems: 'center',
    backgroundColor: Colors.primary, // Hero section için ana renk arka planı
    width: '100%',
  },
  heroTitle: {
    fontSize: Fonts.size.xxxLarge,
    fontWeight: Fonts.weight.bold,
    color: Colors.white,
    textAlign: 'center',
    marginBottom: Layout.spacing.medium,
    lineHeight: Fonts.size.xxxLarge * 1.2,
  },
  heroSubtitle: {
    fontSize: Fonts.size.large,
    color: Colors.white,
    textAlign: 'center',
    marginBottom: Layout.spacing.large,
    opacity: 0.9,
  },
  buttonContainer: {
    width: '100%',
    alignItems: 'center',
    paddingVertical: Layout.spacing.xLarge,
    backgroundColor: Colors.background,
  },
  actionButton: {
    width: '85%', // Buton genişliği
    paddingVertical: Layout.spacing.large,
    borderRadius: Layout.borderRadius.large,
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: Layout.spacing.medium,
    ...Layout.shadow.default, // Genel gölge stilini kullan
  },
  buttonText: {
    color: Colors.white,
    fontSize: Fonts.size.xLarge,
    fontWeight: Fonts.weight.semiBold,
  },
  infoSection: {
    width: '100%',
    paddingHorizontal: Layout.paddingHorizontal,
    paddingVertical: Layout.spacing.xLarge,
    backgroundColor: Colors.surface,
    marginTop: Layout.spacing.large,
    borderTopWidth: 1,
    borderTopColor: Colors.lightGray,
  },
  infoTitle: {
    fontSize: Fonts.size.xLarge,
    fontWeight: Fonts.weight.bold,
    color: Colors.textPrimary,
    marginBottom: Layout.spacing.medium,
    textAlign: 'center',
  },
  infoText: {
    fontSize: Fonts.size.medium,
    color: Colors.textSecondary,
    textAlign: 'center',
    lineHeight: Fonts.size.medium * 1.5,
  },
});