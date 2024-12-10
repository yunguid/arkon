import React, { useState, useEffect } from 'react';
import './DocumentsList.css';

function DocumentsList({ onSelect, refreshTrigger, selectedFileId }) {
  const [documents, setDocuments] = useState([]);
  const [error, setError] = useState(null);

  const fetchDocuments = async () => {
    try {
      const response = await fetch('http://localhost:8000/files');
      if (!response.ok) throw new Error('Failed to fetch documents');
      const data = await response.json();
      setDocuments(data);
      setError(null);
    } catch (err) {
      setError('Failed to load documents');
      console.error('Error fetching documents:', err);
    }
  };

  useEffect(() => {
    fetchDocuments();
  }, [refreshTrigger]);

  return (
    <div className="documents-list">
      <h3>Uploaded Documents</h3>
      {error && <div className="error-message">{error}</div>}
      <ul>
        {documents.map((doc) => (
          <li 
            key={doc.id} 
            onClick={() => onSelect(doc.id)}
            className={selectedFileId === doc.id ? 'selected' : ''}
          >
            <span className="filename">{doc.filename}</span>
            <span className="upload-date">
              {doc.upload_date ? new Date(doc.upload_date).toLocaleDateString() : ''}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default DocumentsList;