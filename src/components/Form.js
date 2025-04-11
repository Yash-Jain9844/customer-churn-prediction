import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import "bootstrap/dist/css/bootstrap.min.css";
import "../index.css";
import {
  User,
  Calendar,
  CreditCard,
  TrendingUp,
  DollarSign,
  ShoppingBag,
  Wallet,
  ShieldCheck,
  Activity,
} from "lucide-react";

export default function Form() {
  const navigate = useNavigate();
  const [step, setStep] = useState(0);

  const questions = [
    {
      id: "creditscore",
      label: "Credit Score",
      type: "number",
      icon: <CreditCard size={20} />,
    },
    { id: "age", label: "Age", type: "number", icon: <Calendar size={20} /> },
    {
      id: "tenure",
      label: "Tenure",
      type: "number",
      icon: <TrendingUp size={20} />,
    },
    {
      id: "balance",
      label: "Balance",
      type: "number",
      icon: <Wallet size={20} />,
    },
    {
      id: "numofproducts",
      label: "Number of Products",
      type: "number",
      icon: <ShoppingBag size={20} />,
    },
    {
      id: "estimatedsalary",
      label: "Estimated Salary",
      type: "number",
      icon: <DollarSign size={20} />,
    },
    {
      id: "gender_Male",
      label: "Gender",
      type: "select",
      icon: <User size={20} />,
      options: [
        { label: "Select", value: "" },
        { label: "Male", value: "true" },
        { label: "Female", value: "false" },
      ],
    },
    {
      id: "hascrcard",
      label: "Has Credit Card",
      type: "select",
      icon: <ShieldCheck size={20} />,
      options: [
        { label: "Select", value: "" },
        { label: "Yes", value: "true" },
        { label: "No", value: "false" },
      ],
    },
    {
      id: "isactivemember",
      label: "Is Active Member",
      type: "select",
      icon: <Activity size={20} />,
      options: [
        { label: "Select", value: "" },
        { label: "Yes", value: "true" },
        { label: "No", value: "false" },
      ],
    },
  ];

  const [formData, setFormData] = useState({
    creditscore: "",
    age: "",
    tenure: "",
    balance: "",
    numofproducts: "",
    estimatedsalary: "",
    gender_Male: "",
    hascrcard: "",
    isactivemember: "",
  });

  const handleChange = (e) => {
    const { id, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [id]:
        id === "gender_Male" || id === "hascrcard" || id === "isactivemember"
          ? value === "true"
          : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const finalData = {
      ...formData,
      creditscore: parseFloat(formData.creditscore),
      age: parseFloat(formData.age),
      tenure: parseFloat(formData.tenure),
      balance: parseFloat(formData.balance),
      numofproducts: parseFloat(formData.numofproducts),
      estimatedsalary: parseFloat(formData.estimatedsalary),
    };

    try {
      const res = await fetch(`${process.env.REACT_APP_API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(finalData),
      });

      if (!res.ok) throw new Error(`HTTP error! Status: ${res.status}`);
      const data = await res.json();
      navigate("/result", { state: { data } });
    } catch (error) {
      console.error("Submission error:", error.message);
    }
  };

  const current = questions[step];

  return (
    <div className="container-fluid py-4 px-3">
      <div
        className="glass-card p-4 w-100 mx-auto"
        style={{ maxWidth: "100%", minWidth: "100%" }}
      >
        <h3 className="text-center fw-bold" style={{ color: "black" }}>
          Customer Churn Predictor
        </h3>

        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label
              htmlFor={current.id}
              className="form-label d-flex align-items-center gap-2"
            >
              {current.icon} {current.label}
            </label>

            {current.type === "select" ? (
              <select
                className="form-select"
                id={current.id}
                value={formData[current.id]}
                onChange={handleChange}
                required
              >
                {current.options.map((opt, i) => (
                  <option key={i} value={opt.value}>
                    {opt.label}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="number"
                className="form-control"
                id={current.id}
                value={formData[current.id]}
                onChange={handleChange}
                required
              />
            )}
          </div>

          <div className="d-flex flex-column flex-sm-row justify-content-between gap-2">
            {step > 0 && (
              <button
                type="button"
                className="btn btn-outline-light w-100"
                onClick={() => setStep(step - 1)}
              >
                Previous
              </button>
            )}
            {step < questions.length - 1 ? (
              <button
                type="button"
                className="btn btn-light w-100"
                onClick={() => setStep(step + 1)}
              >
                Next
              </button>
            ) : (
              <button type="submit" className="btn btn-success w-100">
                Predict
              </button>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}
