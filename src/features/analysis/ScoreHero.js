// src/features/analysis/ScoreHero.js

import React, { useState, useEffect } from 'react'; // useState ve useEffect'i import ediyoruz

function ScoreHero({ score }) {
  // Ekranda görünecek olan skoru tutmak için bir state tanımlıyoruz.
  // Başlangıç değeri 0.
  const [displayScore, setDisplayScore] = useState(0);

  // Bu useEffect, 'score' prop'u her değiştiğinde (yani analiz sonucu geldiğinde) çalışacak.
  useEffect(() => {
    // Eğer hedeflenecek bir skor yoksa, animasyonu başlatma.
    if (score === undefined || score === null) return;

    // Animasyonun toplam süresi (milisaniye cinsinden).
    const animationDuration = 1000; // 1 saniye
    // Her bir artışın ne kadar süreceği.
    const frameDuration = 1000 / 60; // Saniyede 60 frame (daha pürüzsüz animasyon için)
    // Toplam kaç adımda animasyonun biteceği.
    const totalFrames = Math.round(animationDuration / frameDuration);
    // Her bir adımda skora ne kadar ekleneceği.
    const increment = score / totalFrames;

    let currentFrame = 0;
    
    // Her 'frameDuration' süresinde bir sayıyı artıracak olan interval'ı başlat.
    const timer = setInterval(() => {
      currentFrame++;
      const newScore = Math.min(score, Math.round(increment * currentFrame));
      setDisplayScore(newScore);

      // Hedef skora ulaşıldığında interval'ı durdur.
      if (currentFrame === totalFrames) {
        clearInterval(timer);
      }
    }, frameDuration);

    // Bu çok önemli: Eğer bileşen ekrandan kaldırılırsa (örneğin başka sayfaya geçilirse)
    // arkada çalışan interval'ı temizle ki hafıza sızıntısı olmasın.
    return () => clearInterval(timer);

  }, [score]); // Bu effect sadece 'score' prop'u değiştiğinde yeniden çalışır.

  return (
    <div className="score-hero-content">
      <span className="hero-title">Overall Safety Score</span>
      {/* Artık doğrudan 'score' prop'unu değil, animasyonlu 'displayScore' state'ini gösteriyoruz. */}
      <span className="hero-score">{displayScore}</span>
      <span className="hero-tagline">RoomGuard Analysis</span>
    </div>
  );
}

export default ScoreHero;