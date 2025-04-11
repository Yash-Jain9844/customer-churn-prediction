import React from "react";
import { useLocation } from "react-router-dom";
import { CheckCircle, XCircle } from "lucide-react";
import "../result.css";

export default function Results() {
  const location = useLocation();
  const data = location.state?.data;

  if (!data) {
    return (
      <div className="results-container">
        <div className="glass-card">
          <h1>No data available</h1>
        </div>
      </div>
    );
  }

  const { message, prediction } = data;
  const positivePredictions = Object.values(prediction).filter(
    (result) => result === 1
  ).length;

  return (
    <div className="results-container">
      <div className="glass-card">
        <h1>Prediction Results</h1>
        <p>
          <strong>Message:</strong> {message}
        </p>
        <h2>Model Predictions:</h2>
        <table className="results-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Prediction</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(prediction).map(([model, result]) => (
              <tr key={model}>
                <td>{model}</td>
                <td>
                  {result === 1 ? (
                    <CheckCircle color="green" />
                  ) : (
                    <XCircle color="red" />
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="verdict">
          <strong>Verdict:</strong>{" "}
          {positivePredictions >= 2
            ? "The Customer will leave the business."
            : "The Customer will not leave the business."}
        </p>
      </div>
    </div>
  );
}
