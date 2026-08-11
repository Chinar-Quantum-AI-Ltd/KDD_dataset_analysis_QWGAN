import os
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN

def augment_arm_b(X_train, y_train, target_classes, target_ratio=0.3, seeds=[42, 43, 44]):
    """
    Executes Arm B: Generates synthetic datasets using SMOTE and ADASYN 
    across 3 random seeds for specified minority classes.
    """
    augmented_datasets = {}
    
    # Calculate majority class count to determine absolute target sizing
    majority_class = pd.Series(y_train).value_counts().idxmax()
    majority_count = sum(y_train == majority_class)
    
    for seed in seeds:
        print(f"--- Processing Arm B Baselines for Seed {seed} ---")
        
        # Configure sampling strategy ratios dynamically per minority class
        sampling_strategy = {}
        for c in target_classes:
            sampling_strategy[c] = int(majority_count * target_ratio)
            
        # 1. SMOTE Implementation
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=seed)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        
        # 2. ADASYN Implementation
        adasyn = ADASYN(sampling_strategy=sampling_strategy, random_state=seed)
        X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
        
        augmented_datasets[f"smote_seed_{seed}"] = (X_smote, y_smote)
        augmented_datasets[f"adasyn_seed_{seed}"] = (X_adasyn, y_adasyn)
        
    return augmented_datasets
