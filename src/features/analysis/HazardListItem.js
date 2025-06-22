// src/features/analysis/HazardListItem.js

import React from 'react';

function HazardListItem({ hazard, isSelected, onClick }) {
  return (
    <div className={`hazard-list-item ${isSelected ? 'selected' : ''}`} onClick={onClick}>
      <span className={`severity-indicator ${hazard.severity.toLowerCase()}`}></span>
      {hazard.category.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())} Hazard
    </div>
  );
}

export default HazardListItem;