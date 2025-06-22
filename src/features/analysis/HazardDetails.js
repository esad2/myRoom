// src/features/analysis/HazardDetails.js
import React from 'react';
// İkonları import et
import { VscWarning, VscInfo, VscTools, VscLaw } from "react-icons/vsc";

function HazardDetails({ hazard }) {
  if (!hazard) {
    // Bu metin normalde görünmeyecek çünkü çekmece sadece bir tehlike seçildiğinde açılır.
    return null;
  }

  const getSeverityClass = (severity) => {
    if (!severity) return '';
    return severity.toLowerCase();
  }

  return (
    <div className="hazard-details-content">
      <h4><VscWarning className="icon" /> {hazard.category.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())} Hazard</h4>
      <p>
        <strong>Severity:</strong>
        <span className={`severity-tag ${getSeverityClass(hazard.severity)}`}>
          {hazard.severity}
        </span>
      </p>
      <p><VscInfo className="icon" /><strong>Description:</strong> {hazard.description}</p>
      <p><VscTools className="icon" /><strong>Recommended Fix:</strong> {hazard.remediation}</p>
      <p><VscLaw className="icon" /><strong>OSHA Standard:</strong> {hazard.oshaStandard}</p>
    </div>
  );
}

export default HazardDetails;