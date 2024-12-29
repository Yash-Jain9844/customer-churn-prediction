import React from "react";
import { useLocation } from "react-router-dom";

export default function Results() {
  const location = useLocation();
  const data = location.state?.data;

  if (!data) {
    return (
      <div style={styles.container}>
        <h1>No data available</h1>
      </div>
    );
  }
  const { message, prediction } = data;
  const positivePredictions = Object.values(prediction).filter(result => result === 1).length;
  console.log(positivePredictions)
  return (
    <div style={styles.container}>
      <h1 style={styles.heading}>Prediction Results</h1>
      <p style={styles.message}><strong>Message:</strong> {message}</p>
      <h2 style={styles.subHeading}>Model Predictions:</h2>
      <table style={styles.table}>
        <thead>
          <tr>
            <th style={styles.tableHeader}>Model</th>
            <th style={styles.tableHeader}>Prediction</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(prediction).map(([model, result]) => (
            <tr key={model}>
              <td style={styles.tableCell}>{model}</td>
              <td style={styles.tableCell}>{result === 1 ? "Yes" : "No"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p style={styles.verdict}><strong>Verdict: </strong> {positivePredictions >= 2 ? "The Customer will leave the business." : "The Customer will not leave the business."}</p>
    </div>
  );
}

const styles = {
  body:{
    fontFamily: 'Raleway',
  },
  container: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    minHeight: "80vh",
    fontFamily: "Arial, sans-serif",
    textAlign: "center",
    backgroundColor: "rgb(43, 40, 40)",
    padding: "20px",
    fontFamily: 'Raleway',

  },
  heading: {
    fontSize: "2rem",
    marginBottom: "20px",
  },
  verdict:{
    marginTop : "30px",
    fontSize:"2rem"
  },
  message: {
    fontSize: "1.2rem",
    marginBottom: "20px",
  },
  subHeading: {
    fontSize: "1.5rem",
    marginBottom: "10px",
  },
  table: {
    borderCollapse: "collapse",
    width: "80%",
    marginTop: "20px",
  },
  tableHeader: {
    border: "1px solid #ddd",
    padding: "8px",
    // backgroundColor: "white",
    // color: "black",
    textAlign: "center",
  },
  tableCell: {
    border: "1px solid #ddd",
    padding: "8px",
    textAlign: "center",
  },
};
