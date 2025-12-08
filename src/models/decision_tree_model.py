import pandas as pd
from sklearn.tree import DecisionTreeClassifier

features = ["Usage Frequency", "Age", 
            "Support Calls", 
            "Last Interaction", "Total Spend", 
            "Contract Length_Monthly",
            "Gender_Male", "Gender_Female",
            "Last Payment Date_month", "Last Payment Date_day"
            # "Contract Length_Annual", "Contract Length_Quarterly"
            # "Subscription Type_Basic", "Subscription Type_Premium", "Subscription Type_Standard"
            # "Customer Status_inactive", "Customer Status_active"
        ]

depth_limit = 8
columns_to_encode = {
    "Subscription Type": [],
    "Contract Length": [],
    "Gender": [],
    "Customer Status": ["active", "inactive"]
}

# Keep payment delay in, this makes sure that the columns with none or NaN are not removeed.
columns_to_zero_set = ["Support Calls", "Tenure", "Payment Delay", "Last Interaction"] 

columns_to_parse_as_dates = ["Last Due Date", "Last Payment Date"]

def train_decision_tree_model(X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    """Train a model based on a descision tree."""
    model = DecisionTreeClassifier(criterion='entropy', max_depth=depth_limit)
    model.fit(X_train, y_train)
    return model