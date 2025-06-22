// src/features/analysis/InteractiveViewer.js
import React from 'react';
import HazardPopover from './HazardPopover'; // Popover bileşenini import ediyoruz

function InteractiveViewer({ imageUrl, hazards, onMarkerClick, selectedHazard, onImageClick }) {
  const selectedHazardId = selectedHazard?.id;

  // Popover'ı işaretçinin sağında mı solunda mı göstereceğimize karar verelim
  const popoverPositionClass = selectedHazard && selectedHazard.coordinates.x > 70 ? 'left' : 'right';

  return (
    <div className="interactive-viewer" onClick={onImageClick}>
      <img src={imageUrl || 'https://via.placeholder.com/1920x1080.png?text=Image+Not+Found'} alt="Analyzed room" />

      {/* Tehlike İşaretleyicileri */}
      {hazards.map(hazard => (
        <div
          key={hazard.id}
          className={`hazard-marker-wrapper`}
          style={{
            top: `${hazard.coordinates.y}%`,
            left: `${hazard.coordinates.x}%`,
          }}
        >
          <div
            className={`hazard-marker ${selectedHazardId === hazard.id ? 'selected' : ''}`}
            onClick={(e) => {
              e.stopPropagation(); // Arka plana tıklama olayını engelle
              onMarkerClick(hazard);
            }}
            title={hazard.description}
          >
            !
          </div>

          {/* SADECE SEÇİLİ OLAN İŞARETLEYİCİ İÇİN POPOVER GÖSTERİLİR */}
          {selectedHazardId === hazard.id && (
             <HazardPopover hazard={selectedHazard} positionClass={popoverPositionClass} />
          )}
        </div>
      ))}
    </div>
  );
}

export default InteractiveViewer;