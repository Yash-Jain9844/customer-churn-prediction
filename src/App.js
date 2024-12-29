import React, { useState } from "react";
import Form from "./components/Form";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import Output from "./components/Results";

function App() {
  return (
    <>
      <Router>
        <Routes>
          <Route
            exact path="/"
            element={
              <div className="container mt-5">
                <Form></Form>
              </div>
            }
          />
          <Route path="/result" element={<Output />} />
        </Routes>
      </Router>
    </>
  );
}

export default App;
