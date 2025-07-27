from pathlib import Path
import pandas as pd

from src.preprocess import preprocess, create_train_test_split
from src.model import train_model, evaluate_model

if __name__ == "__main__":
    current_dir : Path = Path(__file__)
    project_root : Path = current_dir.parent.parent
    dataPath : Path = project_root / "data" / "diabetes.csv"

    # Load & Preprocess Data
    df : pd.DataFrame = preprocess(str(dataPath))

    # Split Train/Test Data
    X_train, X_test, y_train, y_test = create_train_test_split(df)

    # Train the Model
    model = train_model(X_train, y_train)

    # Evaluate the Model
    evaluate_model(model, X_test, y_test)