import { Dimensions } from 'react-native';

const { width, height } = Dimensions.get('window');

export const Layout = {
  window: {
    width,
    height,
  },
  isSmallDevice: width < 375,

  paddingHorizontal: 20,
  paddingVertical: 15,

  spacing: {
    xSmall: 4,
    small: 8,
    medium: 16,
    large: 24,
    xLarge: 32,
  },

  borderRadius: {
    small: 4,
    medium: 8,
    large: 12,
    circular: 999, // Tamamen yuvarlak butonlar veya avatarlar için
  },

  iconSize: {
    small: 18,
    medium: 24,
    large: 32,
  },

  shadow: {
    default: {
      shadowColor: '#000',
      shadowOffset: {
        width: 0,
        height: 2,
      },
      shadowOpacity: 0.1,
      shadowRadius: 3.84,
      elevation: 5,
    },
  },
};