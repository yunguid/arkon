import React, { useState, useRef } from 'react';
import './UploadForm.css';

const UploadForm = ({ onNewSummary }) => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('idle'); 
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    setLoading(true);
    setError(null);
    setStatus('processing');

    try {
      const res = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || `Upload failed: ${res.status}`);
      }

      onNewSummary(data.summary || null, data.file_id);
      setStatus('complete');
      // Clear input
      setFile(null);
      setFileName('');
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    } catch (err) {
      console.error('Error:', err);
      setError(err.message);
      setStatus('error');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-container">
      <form onSubmit={handleSubmit} className="upload-form">
        <div className="file-input-wrapper">
          <input
            type="file"
            accept=".csv"
            onChange={handleFileChange}
            className="file-input"
            id="file-input"
            ref={fileInputRef}
          />
          <label htmlFor="file-input" className="file-input-label">
            {fileName || 'Choose a CSV file to analyze'}
          </label>
        </div>
        <button
          type="submit"
          className="upload-button"
          disabled={!file || loading}
        >
          {loading ? 'Processing...' : 'Analyze Data'}
        </button>
      </form>

      {status === 'processing' && (
        <div className="loading">
          <div className="loading-spinner"></div>
          <span>Processing your financial data...</span>
        </div>
      )}

      {error && <div className="error-message">{error}</div>}
    </div>
  );
};

export default UploadForm;