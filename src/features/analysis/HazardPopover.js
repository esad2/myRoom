// src/features/analysis/HazardPopover.js
import React from 'react';
import { VscWarning, VscInfo, VscTools, VscLaw, VscFlame } from "react-icons/vsc";

// Kategorilere göre özel ikonlar belirleyelim
const categoryIcons = {
  electrical: <VscWarning />,
  fallAndTrip: <VscWarning />,
  fireEgress: <VscFlame />,
  default: <VscWarning />
};

function HazardPopover({ hazard, positionClass }) {
  if (!hazard) return null;

  const severityClass = hazard.severity.toLowerCase();
  const IconComponent = categoryIcons[hazard.category] || categoryIcons.default;

  return (
    <div className={`hazard-popover ${positionClass}`}>
      <div className={`popover-header ${severityClass}`}>
        {IconComponent}
        <h4>{hazard.category.replace(/([A-Z])/g, ' $1')} Hazard</h4>
      </div>
      <div className="popover-content">
        {/* <p> etiketleri kaldırıldı, her satır artık bir etiket ve bir değerden oluşuyor */}
        <strong>
          <VscInfo className="popover-icon" /> Severity:
        </strong>
        <span className={`severity-text ${severityClass}`}>{hazard.severity}</span>

        <strong>
          <VscInfo className="popover-icon" /> Description:
        </strong>
        <span>{hazard.description}</span>

        <strong>
          <VscTools className="popover-icon" /> Recommended Fix:
        </strong>
        <span>{hazard.remediation}</span>
        
        <strong>
          <VscLaw className="popover-icon" /> OSHA Standard:
        </strong>
        <span>{hazard.oshaStandard}</span>
      </div>
      <div className="popover-arrow"></div>
    </div>
  );
}

export default HazardPopover;