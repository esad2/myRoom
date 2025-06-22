// src/App.js
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import DashboardPage from './pages/DashboardPage';
import './App.css'; // Stil dosyamız

function App() {
  // Gereksiz App-header ve sarmalayıcı div'leri kaldırdık.
  // Artık tüm ekran kontrolü bizim Dashboard sayfamızda.
  return (
    <Router>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/dashboard" element={<DashboardPage />} />
      </Routes>
    </Router>
  );
}

export default App;