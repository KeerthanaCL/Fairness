import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import {
  CloudUpload,
  CheckCircle,
} from 'lucide-react';

const FileUploader = ({ 
  onFileUpload, 
  acceptedTypes = '.csv,.joblib', 
  label = 'Upload File',
  maxSize = 10485760 // 10MB default
}) => {
  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    if (rejectedFiles && rejectedFiles.length > 0) {
      console.error('File rejected:', rejectedFiles[0]?.errors);
      return;
    }

    if (acceptedFiles && acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      
      // Create file data object
      const fileData = {
        name: file.name || 'Unknown file',
        size: file.size || 0,
        type: file.type || '',
        lastModified: file.lastModified || Date.now(),
        file: file,
        // Mock data for demo purposes
        columns: file.name && file.name.includes('.csv') ? [
          'age', 'gender', 'race', 'education', 'income', 
          'employment_status', 'credit_score', 'target'
        ] : null,
        preview: {
          rows: 1000,
          sampleData: []
        },
        validation: {
          isValid: true,
          errors: [],
          warnings: []
        }
      };

      // Call the parent callback
      if (onFileUpload && typeof onFileUpload === 'function') {
        onFileUpload(fileData);
      }
    }
  }, [onFileUpload]);

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    acceptedFiles,
    rejectedFiles
  } = useDropzone({
    onDrop,
    accept: acceptedTypes && typeof acceptedTypes === 'string' 
      ? acceptedTypes.split(',').reduce((acc, type) => {
          acc[type.trim()] = [];
          return acc;
        }, {}) 
      : undefined,
    maxSize,
    multiple: false
  });

  const hasAcceptedFile = acceptedFiles && acceptedFiles.length > 0;
  const hasRejectedFile = rejectedFiles && rejectedFiles.length > 0;

  return (
    <div>
      <div
        {...getRootProps()}
        className={`
          p-6 border-2 border-dashed rounded-lg cursor-pointer text-center transition-all duration-300
          ${isDragActive 
            ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' 
            : 'border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700'
          }
          hover:border-blue-500 hover:bg-blue-50 dark:hover:bg-blue-900/20
        `}
      >
        <input {...getInputProps()} />
        
        {hasAcceptedFile ? (
          <div>
            <CheckCircle className="w-12 h-12 text-green-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              File Uploaded Successfully
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {acceptedFiles[0]?.name} ({(acceptedFiles[0]?.size / 1024 / 1024).toFixed(2)} MB)
            </p>
          </div>
        ) : (
          <div>
            <CloudUpload className="w-12 h-12 text-blue-500 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              {isDragActive ? 'Drop file here' : label}
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              Drag and drop a file here, or click to select
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Accepted types: {acceptedTypes}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-4">
              Max size: {(maxSize / 1024 / 1024).toFixed(0)}MB
            </p>
            <button className="px-4 py-2 border border-blue-500 text-blue-500 rounded-md hover:bg-blue-50 dark:hover:bg-blue-900/20 transition-colors">
              Choose File
            </button>
          </div>
        )}
      </div>

      {hasRejectedFile && rejectedFiles[0] && (
        <div className="mt-4 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
          <p className="text-sm text-red-600 dark:text-red-400">
            File rejected: {rejectedFiles[0].errors?.map(e => e.message).join(', ') || 'Unknown error'}
          </p>
        </div>
      )}
    </div>
  );
};

export default FileUploader;
