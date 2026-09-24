import os
import json
from predict_model import TwoStagePredictor

def main():
    test_dir = r"D:\Hackathon\dataset_processed\pipeline_test_suite"
    
    if not os.path.exists(test_dir):
        os.makedirs(test_dir, exist_ok=True)
        print(f"Created test folder: {test_dir}")
        print("Please place real test images (cars, people, clear leaves, blurry leaves) into this folder and run again.")
        return

    test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not test_files:
        print(f"The test folder {test_dir} is empty.")
        print("Please add images to test the pipeline.")
        return

    predictor = TwoStagePredictor()

    print("==================================================")
    print("Testing Pipeline Scenarios (Real Images)")
    print("==================================================")

    for file in test_files:
        img_path = os.path.join(test_dir, file)
        res = predictor.predict(img_path)
        
        leaf_pred = res.get("leaf_prediction", "N/A")
        leaf_conf = res.get("leaf_confidence", "N/A")
        disease_pred = res.get("disease_prediction", "N/A")
        disease_conf = res.get("disease_confidence", "N/A")
        status = res.get("status", "N/A")
        
        # Format the specific user-requested terminal output
        print(f"\nfilename: {file}")
        print(f"leaf prediction: {leaf_pred}")
        print(f"leaf confidence: {leaf_conf}")
        print(f"disease prediction: {disease_pred}")
        print(f"disease confidence: {disease_conf}")
        print(f"final status: {status.upper()}")

if __name__ == "__main__":
    main()