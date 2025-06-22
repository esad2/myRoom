// src/features/analysis/Scorecard.js
import React from 'react';
import { VscChecklist } from "react-icons/vsc";

function Scorecard({ scores }) {
  return (
    <div className="scorecard">
      <h4 className="panel-subtitle"><VscChecklist /> Safety Scores</h4>
      {Object.entries(scores.categories).map(([key, value]) => (
        <div className="score-item" key={key}>
          <div className="score-item-header">
            <span>{key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase())}</span>
            <span className="score-value">{value} / 100</span>
          </div>
          <div className="score-bar-bg">
            <div
              className="score-bar-fg"
              style={{ width: `${value}%` }}
            ></div>
          </div>
        </div>
      ))}
    </div>
  );
}

export default Scorecard;