// src/features/uploader/FileUploader.js

import React, { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { uploadImageForAnalysis } from '../../services/analysisAPI';
// Ana sayfanın CSS modülünü buradan da import ediyoruz
import styles from '../../pages/HomePage.module.css'; 

function FileUploader() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [fileName, setFileName] = useState('');
  const navigate = useNavigate();
  const fileInputRef = useRef(null);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setFileName(file.name);
      handleUpload(file);
    }
  };

  const handleUpload = async (file) => {
    if (!file) return;
    setIsLoading(true);
    setError('');
    try {
      const analysisResult = await uploadImageForAnalysis(file);
      const imageUrl = URL.createObjectURL(file);
      navigate('/dashboard', { state: { analysisData: analysisResult, uploadedImage: imageUrl } });
    } catch (err) {
      setError('Analysis failed. Please try a different image.');
      setIsLoading(false);
    }
  };

  const onWrapperClick = () => {
    fileInputRef.current.click();
  };

  return (
    <div className={styles.fileUploader}>
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        accept="image/*"
        style={{ display: 'none' }}
      />
      <div className={styles.fileInputWrapper} onClick={onWrapperClick}>
        {isLoading ? (
          <span className={styles.uploadText}>Analyzing...</span>
        ) : fileName ? (
          <span className={styles.fileName}>{fileName}</span>
        ) : (
          <span className={styles.uploadText}>Click or Drag & Drop a Photo Here</span>
        )}
      </div>
      {error && <p className={styles.error}>{error}</p>}
    </div>
  );
}

export default FileUploader;