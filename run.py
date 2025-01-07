import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Step 1: Mock RNA Sequence Processing
def process_rna_sequence(rna_sequence):
    """
    Mock function to simulate gene expression levels from an RNA sequence input.
    """
    np.random.seed(42)
    genes = [f"Gene_{i}" for i in range(1, 101)]  # Mock 100 gene names
    expression_values = np.random.rand(len(genes)) * 10  # Random expression levels
    return pd.DataFrame({"Gene": genes, "Expression": expression_values})

# Step 2: Inject Known Signals for Differential Expression
def inject_known_signals(gene_expression_df, num_significant_genes=10):
    """
    Modify a subset of genes to simulate significant expression differences.
    """
    true_significant_genes = set(np.random.choice(gene_expression_df['Gene'], size=num_significant_genes, replace=False))
    
    group1_expression = gene_expression_df.copy()
    group1_expression['Group'] = 'Group1'
    
    group2_expression = gene_expression_df.copy()
    for gene in true_significant_genes:
        group2_expression.loc[group2_expression['Gene'] == gene, 'Expression'] *= np.random.uniform(1.5, 2.0)
    group2_expression['Group'] = 'Group2'
    
    combined_df = pd.concat([group1_expression, group2_expression])
    return combined_df, true_significant_genes

# Step 3: Perform Differential Expression Analysis
def differential_expression_analysis(combined_df, genes):
    p_values = []
    fold_changes = []
    
    for gene in genes:
        group1 = combined_df[(combined_df['Gene'] == gene) & (combined_df['Group'] == 'Group1')]['Expression']
        group2 = combined_df[(combined_df['Gene'] == gene) & (combined_df['Group'] == 'Group2')]['Expression']
        
        t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
        p_values.append(p_val)
        fold_change = group2.mean() / group1.mean() if group1.mean() != 0 else np.nan
        fold_changes.append(fold_change)
    
    results = pd.DataFrame({
        'Gene': genes,
        'Fold_Change': fold_changes,
        'P_Value': p_values
    })
    results['Significant'] = (results['P_Value'] < 0.05) & (abs(np.log2(results['Fold_Change'])) > 1)
    return results

# Step 4: Train Logistic Regression Model
def train_model(results, true_significant_genes):
    """
    Train a logistic regression model to classify significant genes.
    """
    results['True_Label'] = results['Gene'].apply(lambda x: 1 if x in true_significant_genes else 0)
    
    # Features: log2 Fold Change and -log10 P-Value
    results['log2FC'] = np.log2(results['Fold_Change'].replace(0, np.nan)).fillna(0)
    results['-log10P'] = -np.log10(results['P_Value'].replace(0, np.nan)).fillna(0)
    
    X = results[['log2FC', '-log10P']].values
    y = results['True_Label'].values
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print("\nModel Performance:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model

# Step 5: Main Execution
def main():
    # Mock RNA Sequence Input
    rna_sequence = input("Enter the RNA sequence: ")
    gene_expression_df = process_rna_sequence(rna_sequence)
    
    # Inject Known Signals
    combined_df, true_significant_genes = inject_known_signals(gene_expression_df)
    
    # Differential Expression Analysis
    results = differential_expression_analysis(combined_df, gene_expression_df['Gene'])
    
    # Train and Evaluate Model
    model = train_model(results, true_significant_genes)
    


if __name__ == "__main__":
    main()
