// src/pages/DashboardPage.js
import React, { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { VscArrowLeft } from "react-icons/vsc";

// Bileşenleri import ediyoruz
import InteractiveViewer from '../features/analysis/InteractiveViewer';
import Scorecard from '../features/analysis/Scorecard';
import HazardListItem from '../features/analysis/HazardListItem';
import ScoreHero from '../features/analysis/ScoreHero';

function DashboardPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [analysisData, setAnalysisData] = useState(null);
  const [selectedHazard, setSelectedHazard] = useState(null);

  useEffect(() => {
    if (location.state && location.state.analysisData) {
      setAnalysisData(location.state.analysisData);
      setSelectedHazard(null);
    } else {
      navigate('/');
    }
  }, [location.state, navigate]);

  const handleHazardSelect = (hazard) => {
    // Tıklanan tehlike zaten seçiliyse seçimi kaldırır, değilse onu seçer.
    setSelectedHazard(prev => prev && prev.id === hazard.id ? null : hazard);
  };

  const clearSelection = () => {
    // Boş bir alana tıklandığında seçimi temizler.
    setSelectedHazard(null);
  };

  if (!analysisData) {
    return (
      <div className="loading-container">
        <h2>Loading Analysis...</h2>
      </div>
    );
  }

  const imageUrl = location.state?.uploadedImage;

  return (
    // Popover açıkken `popover-open` class'ı ekleyerek arka planı karartabiliriz.
    <div className={`dashboard-container ${selectedHazard ? 'popover-open' : ''}`}>
      <InteractiveViewer
        imageUrl={imageUrl}
        hazards={analysisData.hazards}
        onMarkerClick={handleHazardSelect}
        selectedHazard={selectedHazard}
        onImageClick={clearSelection} // Resme tıklayınca seçimi temizle
      />

      {/* SOLDAKİ YÜZEN KONTROL PANELİ */}
      <div className="main-panel">
        <button className="back-button" onClick={() => navigate('/')}>
          <VscArrowLeft /> New Scan
        </button>
        <div className="panel-section hero-section">
          <ScoreHero score={analysisData.scores.overall} />
        </div>
        <div className="panel-section">
          <Scorecard scores={analysisData.scores} />
        </div>
        <div className="panel-section list-section">
          <h4 className="panel-subtitle">Detected Hazards</h4>
          <div className="hazard-list-container">
            {analysisData.hazards?.length > 0 ? (
              analysisData.hazards.map(h =>
                <HazardListItem
                  key={h.id}
                  hazard={h}
                  isSelected={selectedHazard?.id === h.id}
                  onClick={() => handleHazardSelect(h)}
                />)
            ) : (
              <p className="no-hazards-message">Congratulations! No hazards were detected.</p>
            )}
          </div>
        </div>
      </div>

      {/* ALTTAN AÇILAN PANEL KALDIRILDI. POPOVER ARTIK InteractiveViewer İÇİNDE. */}
    </div>
  );
}

export default DashboardPage;