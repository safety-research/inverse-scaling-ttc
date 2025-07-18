import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import math
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import numpy as np
import json


import itertools
import re

rcParams = plt.rcParams

def extract_student_id_from_prompt(prompt):
    """Extract the last Student_ID from the prompt field using regex.
    In few-shot examples, multiple Student_IDs appear - we want the last one."""
    matches = re.findall(r'<Student_ID>(.*?)</Student_ID>', prompt)
    if matches:
        return matches[-1].strip()  # Return the last match
    return None

sns.set_theme(style="whitegrid")
rcParams["text.usetex"] = False
rcParams["font.size"] = "12.5"
rcParams["figure.dpi"] = 190
rcParams["font.size"] = "12.5"
rcParams["axes.unicode_minus"] = False
rcParams['font.family'] = 'cmr10'
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.formatter.use_mathtext"] = False
# Set rcParams for the border color and ticks
rcParams["axes.edgecolor"] = "black"  # Set border color
rcParams["axes.linewidth"] = 1.5  # Set border width
rcParams["xtick.color"] = "black"  # Set xtick color
rcParams["ytick.color"] = "black"  # Set ytick color
# set background color
rcParams["axes.facecolor"] = "#F3E7CE"
rcParams["axes.facecolor"] = "#F8EFDE"
# rcParams["axes.facecolor"] = "#EEE7DD"
# rcParams["axes.facecolor"] = "#F8F8F8"
# rcParams["axes.facecolor"] = "#FFFAEB"
rcParams["axes.facecolor"] = "#EDEDED"
rcParams["axes.facecolor"] = "#EFEFEAFF"
# set grid color
rcParams["grid.color"] = "white"
rcParams["grid.alpha"] = 0.7
rcParams["grid.linewidth"] = 1.5
rcParams["grid.linestyle"] = "--"
# make ticks show
rcParams["xtick.bottom"] = True  # Ensure xticks are shown at the bottom
rcParams["ytick.left"] = True  # Ensure yticks are shown on the left
sns.set_context(context="talk", font_scale=0.9)
model_to_model_name = {
    # "O4-mini": "o4-mini",
    # "deepseek-reasoner": "DeepSeek R1",
    "deepseek_r1_0528_awq": "DeepSeek R1",
    "claude-3-7-sonnet-20250219": "Claude Sonnet 3.7",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-opus-4-20250514": "Claude Opus 4",
    "qwen3_8b": "Qwen3 8B",
    "qwen3_14b": "Qwen3 14B",
    "qwen3_32b": "Qwen3 32B",
    "qwq_32b": "QwQ 32B",
    "o3-mini-2025-01-31": "o3-mini",
    "o4-mini-2025-04-16": "o4-mini",
    "o3-2025-04-16": "o3",
    "Ground Truth": "Ground Truth"
}


def main():
    original_dataset_filepath = "data/original/student_lifestyle_regression_dataset.csv"

    # Read the original dataset
    original_dataset = pd.read_csv(original_dataset_filepath)

    # Select relevant columns (keep Student_ID for matching)
    original_dataset = original_dataset[
        [
            "Student_ID",
            "Grades",
            "Study_Hours_Per_Day",
            "Extracurricular_Hours_Per_Day",
            "Sleep_Hours_Per_Day",
            "Social_Hours_Per_Day",
            "Physical_Activity_Hours_Per_Day",
            "Stress_Level",
            "Gender",
        ]
    ]

    # Ensure numeric conversion for non-categorical columns
    numeric_cols = [
        "Grades",
        "Study_Hours_Per_Day",
        "Extracurricular_Hours_Per_Day",
        "Sleep_Hours_Per_Day",
        "Social_Hours_Per_Day",
        "Physical_Activity_Hours_Per_Day",
    ]
    for col in numeric_cols:
        original_dataset[col] = pd.to_numeric(original_dataset[col], errors="coerce")

    # Handle categorical features - Gender
    # Assuming Gender is binary with 1=Male (as specified)
    gender_dummies = pd.get_dummies(original_dataset["Gender"], prefix="Gender")

    # Handle categorical features - Stress_Level
    # Convert stress level to numeric instead of one-hot encoding
    if 'Stress_Level_Numeric' not in original_dataset.columns:
        stress_mapping = {'Low': 0, 'Moderate': 1, 'High': 2}
        original_dataset['Stress_Level_Numeric'] = original_dataset['Stress_Level'].map(stress_mapping)

    # Drop original categorical columns and join the gender dummies (keep Student_ID)
    original_dataset = original_dataset.drop(columns=["Gender", "Stress_Level"])
    original_dataset = pd.concat(
        [original_dataset, gender_dummies], axis=1
    )

    # Read the model results
    # filepath = "results/experiment=claude37__regression/raw/student_lifestyle_regression_Grades_0shot/results.jsonl"
    # filepath = "results-with-scaling/claude-3-7-sonnet-20250219/grades_regression.jsonl"
    # model = "claude-3-7-sonnet-20250219"
    # model = "claude-sonnet-4-20250514"
    # model = "deepseek-reasoner"
    # model = "deepseek_r1_0528_awq"
    # model = "o3-2025-04-16"
    # model = "O4-mini"
    # model = "Qwen3-32B"
    models = [
        "claude-3-7-sonnet-20250219",
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        # "o3-mini-2025-01-31",
        # "o4-mini-2025-04-16",
        # "o3-2025-04-16",
        # # "qwen3_8b",
        # # "qwen3_14b",
        # "qwen3_32b",
        # "qwq_32b",
        # "deepseek_r1_0528_awq",
    ]

    n_shots_setups = [0, 8, 16]

    for model in models:
        for n_shots in n_shots_setups:
            filepath = f"results_v2_scaled/{model}/student_lifestyle_regression_Grades.jsonl"
            model_name = model_to_model_name[model]
            # Store raw JSONL data for analysis
            all_data = []

            model_predictions = {}
            with open(filepath, "r") as f:
                for line in f:
                    data = json.loads(line)
                    reasoning_budget = int(
                        data["reasoning_budget"]
                    )  # Convert to int for sorting later
                    examples_per_prompt = data.get("examples_per_prompt", 0)

                    # Filter for specific n-shot setup
                    if examples_per_prompt != n_shots:
                        continue

                    # Extract Student_ID from prompt to match with original dataset
                    prompt = data.get("prompt", "")
                    student_id = extract_student_id_from_prompt(prompt)
                    if student_id is None:
                        print(f"Could not extract Student_ID from prompt")
                        continue

                    # Store raw data for later analysis
                    all_data.append(data)

                    # Extract answers, handling different formats
                    extracted_answer_original = data["extracted_answer"]
                    if "correct_answer" not in data and "answer" not in data:
                        continue

                    correct_answer = data.get("correct_answer", None)
                    if correct_answer is None:
                        correct_answer = data["answer"]

                    # Try to extract numeric value if answer is in format like "I, 6.9"
                    try:
                        if (
                            isinstance(extracted_answer_original, str)
                            and "," in extracted_answer_original
                        ):
                            extracted_answer = float(
                                extracted_answer_original.split(",")[1].strip()
                            )
                        else:
                            extracted_answer = float(extracted_answer_original)

                        if isinstance(correct_answer, str) and "," in correct_answer:
                            correct_answer = float(correct_answer.split(",")[1].strip())
                        else:
                            correct_answer = float(correct_answer)
                    except (ValueError, IndexError, TypeError):
                        # print(
                        #     f"Could not convert answer to float: {extracted_answer_original}, {correct_answer}"
                        # )
                        continue

                    if reasoning_budget not in model_predictions:
                        model_predictions[reasoning_budget] = {
                            "extracted_answers": [],
                            "correct_answers": [],
                            "student_ids": [],
                        }
                    if extracted_answer > 10:
                        # print(extracted_answer, correct_answer)
                        continue
                    if np.isnan(extracted_answer):
                        print("Answer is NaN:", extracted_answer_original, correct_answer)
                        continue
                    model_predictions[reasoning_budget]["extracted_answers"].append(
                        extracted_answer
                    )
                    model_predictions[reasoning_budget]["correct_answers"].append(
                        correct_answer
                    )
                    model_predictions[reasoning_budget]["student_ids"].append(student_id)

            # Skip if no data for this n-shot setup
            if not model_predictions:
                print(f"No data found for model {model} with {n_shots}-shot setup")
                continue

            print(f"\nProcessing {model_name} - {n_shots}-shot setup")

            # ===== DIAGNOSTIC: Add analysis of extracted answers vs correct answers =====
            print("\n===== DIAGNOSTIC INFORMATION =====")
            for budget in sorted(model_predictions.keys()):
                extracted = np.array(model_predictions[budget]["extracted_answers"])
                correct = np.array(model_predictions[budget]["correct_answers"])

                # Calculate basic statistics
                min_e, max_e = np.min(extracted), np.max(extracted)
                min_c, max_c = np.min(correct), np.max(correct)
                mean_e, mean_c = np.mean(extracted), np.mean(correct)
                std_e, std_c = np.std(extracted), np.std(correct)

                # Calculate correlation between extracted and correct answers
                corr = np.corrcoef(extracted, correct)[0, 1]

                print(f"\nReasoning Budget: {budget}")
                print(f"Number of samples: {len(extracted)}")
                print(
                    f"Extracted answers - Min: {min_e:.2f}, Max: {max_e:.2f}, Mean: {mean_e:.2f}, Std: {std_e:.2f}"
                )
                print(
                    f"Correct answers  - Min: {min_c:.2f}, Max: {max_c:.2f}, Mean: {mean_c:.2f}, Std: {std_c:.2f}"
                )
                print(f"Correlation between extracted and correct answers: {corr:.4f}")

            # Create scatterplots to visualize the relationship
            plt.figure(figsize=(16, 4))
            for i, budget in enumerate(sorted(model_predictions.keys()), 1):
                plt.subplot(1, len(model_predictions), i)
                extracted = model_predictions[budget]["extracted_answers"]
                correct = model_predictions[budget]["correct_answers"]

                # Calculate correlation coefficient
                corr_coef = np.corrcoef(extracted, correct)[0, 1]
                
                # Create scatter plot with correlation-based color
                plt.scatter(correct, extracted, alpha=0.5, edgecolor='black', linewidth=0.5)
                
                # Add identity line
                plt.plot([min(correct), max(correct)], [min(correct), max(correct)], 
                        "r--", label="Identity", linewidth=1.5)
                
                # Add regression line
                z = np.polyfit(correct, extracted, 1)
                p = np.poly1d(z)
                plt.plot(sorted(correct), p(sorted(correct)), 
                        'g-', linewidth=1.5, label=f"Regression (r={corr_coef:.2f})")
                
                # Add title and labels
                plt.title(f"Budget: {budget}\nCorrelation: {corr_coef:.4f}", fontweight='bold')
                plt.xlabel("Ground Truth GPA")
                plt.ylabel("Model Predicted GPA")
                plt.legend(loc='best', fontsize=12)

            plt.tight_layout()
            plt.savefig(f"plots/correlation_gpa/answer_correlation_{model}_{n_shots}.pdf", dpi=300)
            # ===== END DIAGNOSTIC =====

            # Define a mapping for prettier feature names
            feature_name_mapping = {
                "Study_Hours_Per_Day": "Study\n(h/day)",
                "Extracurricular_Hours_Per_Day": "Extracurricular\n(h/day)",
                "Sleep_Hours_Per_Day": "Sleep\n(h/day)",
                "Social_Hours_Per_Day": "Social\n(h/day)",
                "Physical_Activity_Hours_Per_Day": "Physical\nActivity\n(h/day)",
                "Gender_Female": "Female",
                "Gender_Male": "Male",
                "Stress_Level_Numeric": "Stress\nLevel"
            }
            
            # Define mapping for prettier budget labels
            budget_label_mapping = {
                "Ground Truth": "GT",
                **{str(budget): f"{budget}" for budget in sorted(model_predictions.keys())}
            }

            # Initialize list to store correlation results
            correlation_results_list = []

            # Get list of all feature columns after one-hot encoding
            # Excluding "Grades" and "Student_ID" since Grades is our target and Student_ID is for matching
            feature_columns = [col for col in original_dataset.columns if col not in ["Grades", "Student_ID"]]

            # Compute correlations for each feature with the ground truth (Grades)
            for feature in feature_columns:
                # Get feature values
                feature_values = original_dataset[feature].values
                grades_values = original_dataset["Grades"].values

                # Calculate correlation with Grades (Ground Truth)
                ground_truth_corr = np.corrcoef(feature_values, grades_values)[0, 1]

                # Use the pretty feature name if available, otherwise use the original
                pretty_feature_name = feature_name_mapping.get(feature, feature)
                
                correlation_results_list.append(
                    {
                        "Feature": feature,
                        "Pretty_Feature": pretty_feature_name,
                        "Type": "Ground Truth",
                        "Reasoning_Budget": 0,  # Use 0 to sort Ground Truth first
                        "Reasoning_Budget_Label": "Ground Truth",
                        "Pretty_Budget_Label": budget_label_mapping["Ground Truth"],
                        "Correlation": ground_truth_corr,
                    }
                )

                # Calculate correlations with model predictions for each reasoning budget
                for reasoning_budget, predictions in model_predictions.items():
                    student_ids = predictions["student_ids"]
                    extracted_answers = predictions["extracted_answers"]

                    # Match student IDs to get corresponding feature values
                    matched_feature_values = []
                    matched_extracted_answers = []
                    
                    for i, student_id in enumerate(student_ids):
                        # Convert student_id to int to match the CSV data type
                        try:
                            student_id_int = int(student_id)
                        except (ValueError, TypeError):
                            print(f"Warning: Could not convert Student_ID '{student_id}' to integer")
                            continue
                        
                        # Find the corresponding row in original dataset
                        matching_rows = original_dataset[original_dataset["Student_ID"] == student_id_int]
                        if len(matching_rows) == 1:
                            matched_feature_values.append(matching_rows[feature].iloc[0])
                            matched_extracted_answers.append(extracted_answers[i])
                        elif len(matching_rows) == 0:
                            print(f"Warning: Student_ID {student_id_int} not found in original dataset")
                            print(f"  Original student_id from prompt: '{student_id}' (type: {type(student_id)})")
                            print(f"  Sample of Student_IDs in dataset: {original_dataset['Student_ID'].head().tolist()}")
                        else:
                            print(f"Warning: Multiple rows found for Student_ID {student_id_int}")
                            # Use the first match
                            matched_feature_values.append(matching_rows[feature].iloc[0])
                            matched_extracted_answers.append(extracted_answers[i])

                    if len(matched_feature_values) > 0:  # Only calculate if we have matched data
                        extracted_corr = np.corrcoef(
                            matched_feature_values, matched_extracted_answers
                        )[0, 1]

                        budget_label = str(reasoning_budget)
                        pretty_budget_label = budget_label_mapping.get(budget_label, budget_label)
                        
                        correlation_results_list.append(
                            {
                                "Feature": feature,
                                "Pretty_Feature": pretty_feature_name,
                                "Type": "Model Prediction",
                                "Reasoning_Budget": reasoning_budget,  # Store as int for sorting
                                "Reasoning_Budget_Label": budget_label,
                                "Pretty_Budget_Label": pretty_budget_label,
                                "Correlation": extracted_corr,
                            }
                        )
                    else:
                        print(f"Warning: No matched data for feature {feature} and reasoning budget {reasoning_budget}")

            # Convert the list to a DataFrame
            correlation_results = pd.DataFrame(correlation_results_list)

            # Sort by Reasoning_Budget to ensure Ground Truth comes first, then increasing reasoning budgets
            correlation_results = correlation_results.sort_values(by="Reasoning_Budget")
            
            # Filter out unwanted features (Extracurricular, Gender, Social hours)
            features_to_exclude = [
                "Extracurricular\n(h/day)", 
                "Female", 
                "Male", 
                "Social\n(h/day)"
            ]
            correlation_results = correlation_results[~correlation_results["Pretty_Feature"].isin(features_to_exclude)]
            
            # Calculate correlation between true and predicted values for each reasoning budget
            true_vs_predicted_correlations = []
            
            # For Ground Truth, correlation with itself is 1.0
            true_vs_predicted_correlations.append({
                "Feature": "True_vs_Predicted",
                "Pretty_Feature": "Real GPA\nvs. Predicted",  # Special row name
                "Type": "Ground Truth",
                "Reasoning_Budget": 0,
                "Reasoning_Budget_Label": "Ground Truth",
                "Pretty_Budget_Label": "GT",
                "Correlation": 1.0  # Perfect correlation with itself
            })
            
            # For each reasoning budget
            for reasoning_budget, predictions in model_predictions.items():
                true_values = np.array(predictions["correct_answers"])
                predicted_values = np.array(predictions["extracted_answers"])
                
                # Calculate correlation between true and predicted
                corr = np.corrcoef(true_values, predicted_values)[0, 1]
                
                budget_label = str(reasoning_budget)
                pretty_budget_label = budget_label_mapping.get(budget_label, budget_label)
                
                true_vs_predicted_correlations.append({
                    "Feature": "True_vs_Predicted",
                    "Pretty_Feature": "Real GPA\nvs. Predicted",  # Special row name
                    "Type": "Model Prediction",
                    "Reasoning_Budget": reasoning_budget,
                    "Reasoning_Budget_Label": budget_label,
                    "Pretty_Budget_Label": pretty_budget_label,
                    "Correlation": corr
                })
            
            # Add the true vs predicted correlations to the results
            correlation_results = pd.concat([
                correlation_results,
                pd.DataFrame(true_vs_predicted_correlations)
            ])

            # Pivot the DataFrame for heatmap plotting using the pretty names
            correlation_matrix = correlation_results.pivot(
                index="Pretty_Feature", 
                columns="Pretty_Budget_Label", 
                values="Correlation"
            )
            
            # Set specific row order as requested
            row_order = [
                "Real GPA\nvs. Predicted", 
                "Stress\nLevel", 
                "Physical\nActivity\n(h/day)", 
                "Sleep\n(h/day)", 
                "Study\n(h/day)"
            ]
            correlation_matrix = correlation_matrix.reindex(row_order)

            # Reorder columns to ensure GT comes first, then reasoning budgets
            column_order = ["GT"] + [
                f"{budget}" for budget in sorted([b for b in model_predictions.keys()])
            ]
            correlation_matrix = correlation_matrix[column_order]

            # Plot the heatmap with prettier labels
            plt.figure(figsize=(8, 6))  # Slightly larger for better readability
            ax = sns.heatmap(
                correlation_matrix, 
                annot=True, 
                cmap="coolwarm_r", #sns.diverging_palette(10, 130, 67, 53, as_cmap=True), #sns.diverging_palette(10, 130, as_cmap=True),  # Red to green diverging colormap
                center=0, 
                fmt=".2f",
                linewidths=0.5,
                cbar_kws={"label": "Correlation Coefficient"},
            )
            
            # Get the current colorbar and add a black border around it
            cbar = ax.collections[0].colorbar
            cbar.outline.set_edgecolor('black')
            cbar.outline.set_linewidth(2)
            
            # Highlight the "GT vs. Prediction" row with a border
            # Get the position of the "GT vs. Prediction" row (should be first row, index 0)
            model_accuracy_idx = list(correlation_matrix.index).index("Real GPA\nvs. Predicted")
            
            # Add a thicker black border around the "GT vs. Prediction" row
            for i in range(len(correlation_matrix.columns)):
                # Add border to the top of the cell
                ax.add_patch(plt.Rectangle((i, model_accuracy_idx), 1, 0, 
                                    fill=False, edgecolor='black', lw=2))
                # Add border to the bottom of the cell
                ax.add_patch(plt.Rectangle((i, model_accuracy_idx+1), 1, 0, 
                                    fill=False, edgecolor='black', lw=2))
            
            # Add borders to the left and right sides of the entire row
            ax.add_patch(plt.Rectangle((0, model_accuracy_idx), 0, 1, 
                                fill=False, edgecolor='black', lw=2))
            ax.add_patch(plt.Rectangle((len(correlation_matrix.columns), model_accuracy_idx), 0, 1, 
                                fill=False, edgecolor='black', lw=2))
            
            # Add a border around the first column (Ground Truth)
            first_col_idx = 0  # First column index is 0
            # Add vertical borders (left and right sides of the column)
            ax.add_patch(plt.Rectangle((first_col_idx, 0), 0, len(correlation_matrix), 
                                fill=False, edgecolor='black', lw=2))
            ax.add_patch(plt.Rectangle((first_col_idx+1, 0), 0, len(correlation_matrix), 
                                fill=False, edgecolor='black', lw=2))

            
            # Improve title and labels
            plt.title(f"Feature Correlation Grades Regression ({n_shots}-shot) - {model_name}")
            plt.xticks(rotation=45, ha="right")  # Rotate column labels for better readability
            plt.yticks(rotation=0)  # Keep row labels horizontal
            
            # Remove axis labels
            ax.set_xlabel("")  # Remove x-axis label
            ax.set_ylabel("")  # Remove y-axis label
            
            # Adjust layout
            plt.tight_layout()
            
            # Add a border around the entire plot
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color('black')
                spine.set_linewidth(2)
            

            # Save the figure as a PNG with higher resolution
            plt.savefig(f"plots/correlation_gpa/correlation_heatmap_{model}_{n_shots}shot.pdf", dpi=300, bbox_inches="tight")
            # plt.show()


if __name__ == "__main__":
    main()
