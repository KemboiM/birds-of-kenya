import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.data.loader import prepare_dataset
from src.models.train import train_model, evaluate_model
from src.utils.helpers import save_metrics, log_training

def main():
    """Main pipeline for training bird classification model."""
    
    # Configuration
    SPECIES_NAME = "african-fish-eagle"
    DATA_DIR = f"data/raw/{SPECIES_NAME}"
    EPOCHS = 10
    BATCH_SIZE = 16
    
    print("\n" + "="*60)
    print(f"Bird Classifier: {SPECIES_NAME.upper()}")
    print("="*60)
    
    # Check if data exists
    print(f"\nLooking for images in: {DATA_DIR}")
    if not os.path.exists(DATA_DIR):
        print("ERROR: Directory not found!")
        print(f"Create it with: mkdir -p {DATA_DIR}")
        print("Then add your bird images to that folder")
        return
    
    # Step 1: Load dataset
    print("\nStep 1: Loading images...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset(DATA_DIR)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Step 2: Train model
    print("\nStep 2: Training model...")
    try:
        model, history = train_model(X_train, y_train, X_val, y_val, epochs=EPOCHS, batch_size=BATCH_SIZE)
    except Exception as e:
        print(f"Error training model: {e}")
        return
    
    # Step 3: Evaluate
    print("\nStep 3: Evaluating model...")
    try:
        metrics = evaluate_model(model, X_test, y_test)
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return
    
    # Step 4: Save results
    print("\nStep 4: Saving results...")
    save_metrics(metrics, SPECIES_NAME)
    log_training(SPECIES_NAME, EPOCHS, BATCH_SIZE, metrics)
    
    print("\n" + "="*60)
    print("Training complete!")
    print("Model saved to: models/best_model.h5")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
