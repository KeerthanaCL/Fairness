import React, { useState } from 'react';
import { ChevronDown, X } from 'lucide-react';

const SensitiveAttributeSelector = ({ 
  columns = [], 
  selectedAttributes = [], 
  onAttributesChange 
}) => {
  const [isOpen, setIsOpen] = useState(false);

  const handleAttributeToggle = (attribute) => {
    const newAttributes = selectedAttributes.includes(attribute)
      ? selectedAttributes.filter(attr => attr !== attribute)
      : [...selectedAttributes, attribute];
    onAttributesChange(newAttributes);
  };

  const removeAttribute = (attributeToRemove) => {
    const newAttributes = selectedAttributes.filter(attr => attr !== attributeToRemove);
    onAttributesChange(newAttributes);
  };

  // Filter columns that are likely to be sensitive attributes
  const potentialSensitiveAttributes = columns.filter(col => 
    ['gender', 'sex', 'race', 'ethnicity', 'age', 'religion', 'disability', 'sexual_orientation']
      .some(sensitive => col.toLowerCase().includes(sensitive))
  );

  return (
    <div className="relative">
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
        Sensitive Attributes
      </label>
      
      {/* Selected attributes display */}
      <div className="min-h-[42px] p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 mb-2">
        <div className="flex flex-wrap gap-1">
          {selectedAttributes.map((attribute) => (
            <span 
              key={attribute} 
              className="inline-flex items-center gap-1 px-2 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200 text-xs rounded-md"
            >
              {attribute}
              <button
                onClick={() => removeAttribute(attribute)}
                className="hover:bg-blue-200 dark:hover:bg-blue-800 rounded-full p-0.5"
              >
                <X className="w-3 h-3" />
              </button>
            </span>
          ))}
          {selectedAttributes.length === 0 && (
            <span className="text-gray-400 dark:text-gray-500 text-sm">
              Select sensitive attributes...
            </span>
          )}
        </div>
      </div>

      {/* Dropdown toggle */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-left flex items-center justify-between hover:border-blue-500 focus:border-blue-500 focus:ring-1 focus:ring-blue-500"
      >
        <span className="text-sm text-gray-700 dark:text-gray-300">
          Select from available columns
        </span>
        <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {/* Dropdown menu */}
      {isOpen && (
        <div className="absolute z-10 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md shadow-lg max-h-60 overflow-y-auto">
          {columns.map((column) => {
            const isSelected = selectedAttributes.includes(column);
            const isSuggested = potentialSensitiveAttributes.includes(column);
            
            return (
              <button
                key={column}
                onClick={() => handleAttributeToggle(column)}
                className={`
                  w-full px-3 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center justify-between
                  ${isSelected ? 'bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300' : 'text-gray-700 dark:text-gray-300'}
                  ${isSuggested ? 'font-semibold' : 'font-normal'}
                `}
              >
                <span>
                  {column}
                  {isSuggested && ' (Suggested)'}
                </span>
                {isSelected && (
                  <span className="text-blue-500">✓</span>
                )}
              </button>
            );
          })}
          {columns.length === 0 && (
            <div className="px-3 py-2 text-sm text-gray-500 dark:text-gray-400">
              No columns available
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default SensitiveAttributeSelector;
