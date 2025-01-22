import React, { useState } from 'react';

const DifferentialPanel = () => {
  const [diagnoses, setDiagnoses] = useState([
    { name: "Choledocholithiasis", notes: "Common cause of obstructive jaundice" },
    { name: "Acute Cholangitis", notes: "Consider if fever develops" },
    { name: "Pancreatic Cancer", notes: "Less likely given age but needs consideration" }
  ]);

  return (
    <div className="space-y-2">
      {diagnoses.map((dx, index) => (
        <div key={dx.name} className="flex items-start space-x-2">
          {/* Arrows column */}
          <div className="flex flex-col justify-center space-y-1 pt-2">
            {index > 0 && (
              <button className="p-1 hover:bg-gray-100 rounded">
                <span className="text-gray-600">▲</span>
              </button>
            )}
            {index < diagnoses.length - 1 && (
              <button className="p-1 hover:bg-gray-100 rounded">
                <span className="text-gray-600">▼</span>
              </button>
            )}
          </div>
          
          {/* Main content column */}
          <div className="flex-1">
            <div className="border rounded-lg">
              <div className="p-2 bg-gray-50 rounded-t-lg">
                <span className="font-medium">#{index + 1}: {dx.name}</span>
              </div>
              <div className="p-2 border-t">
                {dx.notes}
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default DifferentialPanel;