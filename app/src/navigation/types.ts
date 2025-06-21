export type RootStackParamList = {
  Home: undefined;
  Upload: undefined;
  RoomDetail: { roomId: string }; // Oda detayları için bir roomId alacak
  Reports: undefined;
  // Gelecekte eklenecek diğer ekranlar...
};

// React Navigation'a özel tipleri de burada tanımlayabiliriz
declare global {
  namespace ReactNavigation {
    interface RootParamList extends RootStackParamList {}
  }
}