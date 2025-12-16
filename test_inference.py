"""
Test script to verify both saved airline satisfaction models
can be loaded and used for inference.
"""

import joblib
import pandas as pd
from pathlib import Path


def test_model(model_path):
    """Load model and test inference."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_path}")
    print('=' * 60)

    try:
        # Load model
        print("Loading model...", end=" ")
        model = joblib.load(model_path)
        print("✓")

        if hasattr(model, "named_steps"):
            print(f"Pipeline steps: {list(model.named_steps.keys())}")

        # Sample data based on Airline Passenger Satisfaction dataset
        sample_data = pd.DataFrame({
            "Gender": ["Female", "Male", "Female"],
            "Customer Type": ["Loyal Customer", "Disloyal Customer", "Loyal Customer"],
            "Age": [29, 45, 34],
            "Type of Travel": ["Business travel", "Personal Travel", "Business travel"],
            "Class": ["Business", "Eco", "Eco Plus"],
            "Flight Distance": [1200, 800, 1500],

            "Inflight wifi service": [4, 2, 5],
            "Departure/Arrival time convenient": [3, 4, 4],
            "Ease of Online booking": [4, 3, 5],
            "Gate location": [3, 2, 4],
            "Food and drink": [5, 2, 4],
            "Online boarding": [5, 3, 4],
            "Seat comfort": [5, 3, 4],
            "Inflight entertainment": [5, 2, 5],
            "On-board service": [5, 3, 4],
            "Leg room service": [4, 3, 5],
            "Baggage handling": [5, 4, 4],
            "Checkin service": [4, 3, 5],
            "Inflight service": [5, 3, 5],
            "Cleanliness": [5, 3, 4],

            "Departure Delay in Minutes": [10, 45, 5],
            "Arrival Delay in Minutes": [5, 60, 0]
        })

        # Predict
        print("Running inference...", end=" ")
        predictions = model.predict(sample_data)
        print("✓")

        # Display results
        print("\nPredictions:")
        for i, pred in enumerate(predictions, 1):
            label = "Satisfied" if pred == 1 else "Neutral or Dissatisfied"
            print(f"  Sample {i}: {label}")

        print(f"\n✓ {Path(model_path).name} - SUCCESS")
        return True

    except Exception as e:
        print(f"\n✗ FAILED: {e}")
        return False


def main():
    """Test both airline satisfaction models."""
    models = [
        "models/global_best_model.pkl",
        "models/global_best_model_optuna.pkl"
    ]

    print("Testing both models...")
    results = [test_model(m) for m in models]

    print(f"\n{'=' * 60}")
    print(f"SUMMARY: {sum(results)}/{len(results)} models passed")
    print('=' * 60)

    return all(results)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
