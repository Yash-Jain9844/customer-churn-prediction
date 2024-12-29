import React, { useState } from "react";
import { useNavigate } from "react-router-dom"
;
export default function Demo() {
    const navigate = useNavigate(); // Initialize the navigate function

  const [formData, setFormData] = useState({
    // creditscore: 222,
    age: 34,
//    tenure: 5,
    balance: 1000.0,
    numofproducts: 2,
  //  estimatedsalary: 100000.0,
    geography_Germany: true,
    //geography_Spain: false,
    //gender_Male: true,
    //hascrcard: true,
    isactivemember: false,
  });

  const handleChange = (e) => {
    const { id, value } = e.target;
    if (id === "Geography") {
      setFormData({
        ...formData,
        geography_Germany: value === "Germany",
        geography_Spain: value === "Spain",
      });
    } else if (
      id === "gender_Male" ||
      id === "hascrcard" ||
      id === "isactivemember"
    ) {
      setFormData({ ...formData, [id]: value === "true" });
    } else if (
      [
        "creditscore",
        "age",
        "tenure",
        "balance",
        "numofproducts",
        "estimatedsalary",
      ].includes(id)
    ) {
      setFormData({ ...formData, [id]: parseFloat(value) });
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("Form Data Submitted:", formData);

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!res.ok) {
        throw new Error(`HTTP error! Status: ${res.status}`);
      }

      const data = await res.json();
      console.log("Server Response:", data);
      navigate("/result", { state: { data } }); // Pass data via state
    } catch (error) {
      console.error("Error during submission:", error.message || error);
    }
  };
  return (
    <>
      <h2>Customer Churn Predictor</h2>
      <form onSubmit={handleSubmit}>
        {/* Credit Score */}
        <div className="mb-3">
          <label htmlFor="creditscore" className="form-label">
            Credit Score :{" "}
          </label>
          <input
            type="number"
            className="form-control"
            id="creditscore"
            onChange={handleChange}
            placeholder="Enter credit score"
          />
        </div>

        {/* age */}
        <div className="mb-3">
          <label htmlFor="age" className="form-label">
            Age :{" "}
          </label>
          <input
            type="number"
            className="form-control"
            id="age"
            // value={formData.age}
            onChange={handleChange}
            placeholder="Enter age"
          />
        </div>

        {/* tenure */}
        <div className="mb-3">
          <label htmlFor="tenure" className="form-label">
            Tenure :{" "}
          </label>
          <input
            type="number"
            className="form-control"
            id="tenure"
            // value={formData.tenure}
            onChange={handleChange}
            placeholder="Enter tenure"
          />
        </div>

        {/* balance */}
        <div className="mb-3">
          <label htmlFor="balance" className="form-label">
            Balance :{" "}
          </label>
          <input
            type="number"
            className="form-control"
            id="balance"
            // value={formData.balance}
            onChange={handleChange}
            placeholder="Enter balance"
          />
        </div>

        {/* Number of Products */}
        <div className="mb-3">
          <label htmlFor="numofproducts" className="form-label">
            Number of Products :{" "}
          </label>
          <input
            type="number"
            className="form-control"
            id="numofproducts"
            // value={formData.numofproducts}
            onChange={handleChange}
            placeholder="Enter number of products"
          />
        </div>

        {/* Estimated Salary */}
        <div className="mb-3">
          <label htmlFor="estimatedsalary" className="form-label">
            Estimated Salary :{" "}
          </label>
          <input
            type="number"
            className="form-control"
            id="estimatedsalary"
            // value={formData.estimatedsalary}
            onChange={handleChange}
            placeholder="Enter estimated salary"
          />
        </div>

        {/* Geography */}
        <div className="mb-3">
          <label htmlFor="Geography" className="form-label">
            Geography :{" "}
          </label>
          <select
            className="form-select"
            id="Geography"
            onChange={handleChange}
          >
            <option value="">Select</option>
            <option value="Germany">Germany</option>
            <option value="Spain">Spain</option>
          </select>
        </div>

        {/* Gender */}
        <div className="mb-3">
          <label htmlFor="gender_Male" className="form-label">
            Gender :{" "}
          </label>
          <select
            className="form-select"
            id="gender_Male"
            onChange={handleChange}
          >
            <option value="">Select</option>
            <option value="true">Male</option>
            <option value="false">Female</option>
          </select>
        </div>

        {/* Has Credit Card */}
        <div className="mb-3">
          <label htmlFor="hascrcard" className="form-label">
            Has Credit Card :{" "}
          </label>
          <select
            className="form-select"
            id="hascrcard"
            onChange={handleChange}
          >
            <option value="">Select</option>
            <option value="true">Yes</option>
            <option value="false">No</option>
          </select>
        </div>

        {/* Is Active Member */}
        <div className="mb-3">
          <label htmlFor="isactivemember" className="form-label">
            Is Active Member :{" "}
          </label>
          <select
            className="form-select"
            id="isactivemember"
            onChange={handleChange}
          >
            <option value="">Select</option>
            <option value="true">Yes</option>
            <option value="false">No</option>
          </select>
        </div>

        {/* Submit Button */}
        <div className="d-flex justify-content-center">
          <button
            style={{ width: "50%" }}
            type="submit"
            className="btn btn-primary"
          >
            Predict :{" "}
          </button>
        </div>
      </form>
    </>
  );
}
