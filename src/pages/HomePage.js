// src/pages/HomePage.js

import React from 'react';
import FileUploader from '../features/uploader/FileUploader';
import styles from './HomePage.module.css'; // Yeni CSS modülünü import ediyoruz

function HomePage() {
  return (
    // CSS dosyasındaki class'ları 'styles' objesi üzerinden kullanıyoruz
    <div className={styles.container}>
      <h1 className={styles.title}>RoomGuard</h1>
      <p className={styles.subtitle}>
        Analyze your room's safety in seconds. Upload a photo to identify potential hazards and receive your comprehensive safety score.
      </p>
      <FileUploader />
    </div>
  );
}

export default HomePage;