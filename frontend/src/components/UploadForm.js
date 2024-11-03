import React, { useState } from 'react';

function UploadForm() {
  const [file, setFile] = useState(null);
  const [insights, setInsights] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch('http://localhost:8000/upload', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    setInsights(data.insights);
  };

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          type="file"
          accept=".csv"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button type="submit">Upload and Analyze</button>
      </form>
      {insights && (
        <div>
          <h2>AI Insights</h2>
          <p>{insights}</p>
        </div>
      )}
    </div>
  );
}

export default UploadForm; 