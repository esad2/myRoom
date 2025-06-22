// src/services/analysisAPI.js

import axios from 'axios';

// Backend sunucumuzun adresi
const API_URL = 'http://localhost:3001/api/analyze';

export const uploadImageForAnalysis = async (file) => {
  // Fotoğrafı bir form verisi olarak hazırlıyoruz
  const formData = new FormData();
  formData.append('image', file); // 'image' ismi, backend'deki upload.single('image') ile eşleşmeli

  console.log("Sending image to the REAL backend...");

  // Axios ile backend'imize POST isteği atıyoruz
  const response = await axios.post(API_URL, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  // Backend'den gelen JSON verisini doğrudan döndürüyoruz
  return response.data;
};