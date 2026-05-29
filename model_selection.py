import os
import time
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from output.export_pubfig import export_pubfig
from core.config import _BASE_DIR


# =========================
# FIXED CLASS ORDER (IMPORTANT)
# =========================
GESTURE_ORDER = [
    "PalmarPinch",
    "2FingerPinch",
    "LateralPinch",
    "Wrap",
    "Open",
    "Relaxed"
]

LABEL_MAP = {g: i for i, g in enumerate(GESTURE_ORDER)}
INV_LABEL_MAP = {i: g for g, i in LABEL_MAP.items()}


def adaptivePCA_classification(
    dataFile,
    n_components_list,
    test_size=0.3,
    classifiers=None,
    training_mode='shuffled',
    n_training_samples=None,
    run_id=None,
    test_run_ids=None
):

    if classifiers is None:
        classifiers = ["k-NN", "Centroid", "GMM", "Mahalanobis", "LDA", "SVM"]

    df = pd.read_csv(dataFile)

    # =========================
    # FORCE CONSISTENT LABELS
    # =========================
    df = df[df['gesture'].isin(GESTURE_ORDER)].copy()
    df['label'] = df['gesture'].map(LABEL_MAP)

    gesture_names = np.array(GESTURE_ORDER)

    # Separate training/testing runs
    if test_run_ids is not None:
        print(f"\n[Cross-Run Testing] Train on run {run_id}, test on runs {test_run_ids}")
        df_train = df[df['run'] == run_id] if run_id is not None else df
        df_test = df[df['run'].isin(test_run_ids)]

        if len(df_train) == 0:
            raise ValueError(f"No training data found for run {run_id}")
        if len(df_test) == 0:
            raise ValueError(f"No test data found for runs {test_run_ids}")
    else:
        if run_id is not None and training_mode == 'structured':
            print(f"\n[Run Filter] Using only run: {run_id}")
            df = df[df['run'] == run_id]
        df_train = df
        df_test = None

    # Training sample selection
    if n_training_samples is not None:
        if training_mode == 'shuffled':
            df_train = df_train.sample(
                n=min(n_training_samples, len(df_train)),
                random_state=42
            )

        elif training_mode == 'structured':
            df_list = []
            for gesture in gesture_names:
                gesture_df = df_train[df_train['gesture'] == gesture]
                n_samples = min(n_training_samples, len(gesture_df))
                if n_samples > 0:
                    df_list.append(gesture_df.sample(n=n_samples, random_state=42))
            df_train = pd.concat(df_list, ignore_index=True)

    results_table = pd.DataFrame(
        index=pd.MultiIndex.from_product([classifiers, n_components_list]),
        columns=['Analysis'],
        dtype=float
    )

    best_predictions = {}

    for n_components in n_components_list:
        print(f"\n--- n_components = {n_components} ---")

        if test_run_ids is not None:
            X_train = df_train.drop(['gesture', 'run', 'iteration', 'label'], axis=1).values
            y_train = df_train['label'].values

            X_test = df_test.drop(['gesture', 'run', 'iteration', 'label'], axis=1).values
            y_test = df_test['label'].values
        else:
            data = df_train.drop(['gesture', 'run', 'iteration', 'label'], axis=1).values
            labels = df_train['label'].values

            X_train, X_test, y_train, y_test = train_test_split(
                data, labels,
                test_size=test_size,
                random_state=42,
                stratify=labels
            )

        # Standardization
        scaler = StandardScaler()
        X_train_S = scaler.fit_transform(X_train)
        X_test_S = scaler.transform(X_test)

        # PCA
        max_components = min(n_components, X_train_S.shape[0] - 1)

        pca = PCA(n_components=max_components)
        pca.fit(X_train_S)

        X_train_pca = pca.transform(X_train_S)
        X_test_pca = pca.transform(X_test_S)

        predictions_dict = {}
        accuracies = {}

        # =========================
        # CLASSIFIERS
        # =========================

        if "k-NN" in classifiers:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn.fit(X_train_pca, y_train)
            pred = knn.predict(X_test_pca)
            predictions_dict["k-NN"] = pred
            accuracies["k-NN"] = np.mean(pred == y_test)
            results_table.loc[("k-NN", n_components), 'Analysis'] = accuracies["k-NN"]

        if "Centroid" in classifiers:
            nc = NearestCentroid()
            nc.fit(X_train_pca, y_train)
            pred = nc.predict(X_test_pca)
            predictions_dict["Centroid"] = pred
            accuracies["Centroid"] = np.mean(pred == y_test)
            results_table.loc[("Centroid", n_components), 'Analysis'] = accuracies["Centroid"]

        if "GMM" in classifiers:
            gmm_models = []
            for i in range(len(GESTURE_ORDER)):
                Xg = X_train_pca[y_train == i]
                if len(Xg) >= 2:
                    gmm = GaussianMixture(n_components=1, covariance_type='full')
                    gmm.fit(Xg)
                    gmm_models.append(gmm)
                else:
                    gmm_models.append(None)

            log_likelihoods = np.zeros((len(X_test_pca), len(GESTURE_ORDER)))

            for i, gmm in enumerate(gmm_models):
                if gmm is not None:
                    log_likelihoods[:, i] = gmm.score_samples(X_test_pca)
                else:
                    log_likelihoods[:, i] = -np.inf

            pred = np.argmax(log_likelihoods, axis=1)
            predictions_dict["GMM"] = pred
            accuracies["GMM"] = np.mean(pred == y_test)
            results_table.loc[("GMM", n_components), 'Analysis'] = accuracies["GMM"]

        if "Mahalanobis" in classifiers:
            means = []
            covs = []

            for i in range(len(GESTURE_ORDER)):
                Xg = X_train_pca[y_train == i]
                means.append(np.mean(Xg, axis=0))
                covs.append(np.cov(Xg.T) + 1e-6 * np.eye(Xg.shape[1]))

            means = np.array(means)

            dists = np.zeros((len(X_test_pca), len(GESTURE_ORDER)))

            for i in range(len(GESTURE_ORDER)):
                diff = X_test_pca - means[i]
                inv_cov = np.linalg.inv(covs[i])
                dists[:, i] = np.sum(diff @ inv_cov * diff, axis=1)

            pred = np.argmin(dists, axis=1)
            predictions_dict["Mahalanobis"] = pred
            accuracies["Mahalanobis"] = np.mean(pred == y_test)
            results_table.loc[("Mahalanobis", n_components), 'Analysis'] = accuracies["Mahalanobis"]

        if "LDA" in classifiers:
            lda = LDA(n_components=min(2, len(GESTURE_ORDER) - 1))
            lda.fit(X_train_pca, y_train)
            pred = lda.predict(X_test_pca)

            predictions_dict["LDA"] = pred
            accuracies["LDA"] = np.mean(pred == y_test)
            results_table.loc[("LDA", n_components), 'Analysis'] = accuracies["LDA"]

        if "SVM" in classifiers:
            model = svm.SVC(kernel='linear', max_iter=1000)
            model.fit(X_train_pca, y_train)
            pred = model.predict(X_test_pca)

            predictions_dict["SVM"] = pred
            accuracies["SVM"] = np.mean(pred == y_test)
            results_table.loc[("SVM", n_components), 'Analysis'] = accuracies["SVM"]

        # Best classifier
        best_clf = max(accuracies, key=accuracies.get)
        best_predictions[n_components] = {
            "classifier": best_clf,
            "accuracy": accuracies[best_clf],
            "predictions": predictions_dict[best_clf],
            "y_test": y_test,
            "gesture_names": gesture_names
        }

        # =========================
        # PLOTTING (UNCHANGED LOGIC)
        # =========================
        if n_components <= 3:
            cmap = plt.get_cmap('tab20', len(GESTURE_ORDER))

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d') if n_components == 3 else plt.gca()

            for i, g in enumerate(GESTURE_ORDER):
                idx = (y_train == i)
                idx_test = (y_test == i)

                if n_components == 2:
                    ax.scatter(X_train_pca[idx, 0], X_train_pca[idx, 1], color=cmap(i))
                    ax.scatter(X_test_pca[idx_test, 0], X_test_pca[idx_test, 1], marker='x', color=cmap(i))
                else:
                    ax.scatter(X_train_pca[idx, 0], X_train_pca[idx, 1], X_train_pca[idx, 2], color=cmap(i))
                    ax.scatter(X_test_pca[idx_test, 0], X_test_pca[idx_test, 1], X_test_pca[idx_test, 2], marker='x', color=cmap(i))

            ax.set_title(f"n_components={n_components}")
            plt.show()

    return results_table, best_predictions

if __name__ == "__main__":
    # DATA_PATH = r"C:\Users\julm\OneDrive - Aalborg Universitet\Master thesis\Code\Module\RMGrealtime\output\analysis_data d200 20x5p 3r.csv"
    DATA_PATH = _BASE_DIR / "output" / "analysis_data d200 20x5p 3r.csv"
    n_components_list = [10, 15, 20, 25, 40, 60]
    
    # Customize which classifiers to test (remove any from this list to exclude them)
    selected_classifiers = ["k-NN","Centroid", "GMM", "Mahalanobis", "LDA", "SVM"]
    
    # === TRAINING DATA SELECTION ===
    # Option 1: Use all data (default - 70/30 split)
    # results_df, best_predictions = adaptivePCA_classification(
    #     DATA_PATH, n_components_list, test_size=0.3, classifiers=selected_classifiers
    # )
        
    # Option 3: Structured mode - 30 samples per gesture from ALL runs (70/30 split)
    results_df, best_predictions = adaptivePCA_classification(
        DATA_PATH, n_components_list, test_size=0.3, classifiers=selected_classifiers,
        training_mode='structured', n_training_samples=60, test_run_ids=[0,1,2,3,4,5,6,7,8,9]
    )
    
    # === CROSS-RUN TESTING (NEW) ===
    # Train on run 1, test on runs 2-9
    
    # Option 6: 30 samples per gesture from run 1 (180 samples), test on other runs
    # results_df, best_predictions = adaptivePCA_classification(
    #     DATA_PATH, n_components_list, classifiers=selected_classifiers,
    #     training_mode='structured', n_training_samples=60, run_id=0, 
    #     test_run_ids=[0,1,2]
    # )
    
    print("\n=== Classification Accuracy Table ===")
    print(results_df)
    
    # Save results to TXT (human-readable)
    # txt_path = r"C:\Users\julm\OneDrive - Aalborg Universitet\Master thesis\Code\Module\RMGrealtime\output\classification_results.txt"
    txt_path = _BASE_DIR / "output" / "classification_results.txt"
    with open(txt_path, 'w') as f:
        f.write("=== Classification Accuracy Results ===\n\n")
        f.write(results_df.to_string())
    print("[Results saved to TXT]")
    
    # Plot accuracy vs dimensions for each classifier
    fig, ax = plt.subplots(figsize=(10, 6))
    
    classifiers = results_df.index.get_level_values(0).unique()
    for clf in classifiers:
        accuracies = results_df.loc[clf, 'Analysis'].values
        ax.plot(n_components_list, accuracies, marker='o', label=clf, linewidth=2, markersize=8)
    
    ax.set_xlabel('Number of PCA Components', fontsize=12)
    ax.set_ylabel('Classification Accuracy', fontsize=12)
    ax.set_title('Classifier Accuracy vs PCA Dimensions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # plot_path = r"C:\Users\julm\OneDrive - Aalborg Universitet\Master thesis\Code\Module\RMGrealtime\output\accuracy_vs_dimensions.pdf"
    plot_path = _BASE_DIR / "output" / "accuracy_vs_dimensions.pdf"
    # plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    export_pubfig(fig, plot_path, width=10, aspectRatio=9/16)
        
    print(f"✓ Plot saved to: {plot_path}")
    plt.show()
    
    # Generate confusion matrices for best classifier at each dimension
    print("\n=== Generating Confusion Matrices ===")
    from sklearn.metrics import ConfusionMatrixDisplay
    from matplotlib.colors import LogNorm
    
    n_dims = len(n_components_list)
    n_cols = min(3, n_dims)
    n_rows = (n_dims + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_dims == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, n_comp in enumerate(n_components_list):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        best_data = best_predictions[n_comp]
        clf_name = best_data["classifier"]
        predictions = best_data["predictions"]
        y_true = best_data["y_test"]
        gesture_names = best_data["gesture_names"]
        accuracy = best_data["accuracy"]
        
        # Create confusion matrix with normalized probabilities
        cm = confusion_matrix(y_true, predictions)
        cm_prob = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot with log scale for better visibility of misclassifications
        im = ax.imshow(cm_prob, cmap='Blues', norm=LogNorm(vmin=cm_prob[cm_prob > 0].min(), vmax=1.0))
        
        # Add text annotations showing probabilities
        for i in range(len(gesture_names)):
            for j in range(len(gesture_names)):
                text = ax.text(j, i, f'{cm_prob[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=9)
        
        ax.set_xticks(np.arange(len(gesture_names)))
        ax.set_yticks(np.arange(len(gesture_names)))
        ax.set_xticklabels(gesture_names, rotation=45, ha='right')
        ax.set_yticklabels(gesture_names)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(f'n_comp={n_comp}: {clf_name} (Acc: {accuracy:.3f})', fontweight='bold')
    
    # Hide unused subplots
    for idx in range(n_dims, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    # cm_path = r"C:\Users\julm\OneDrive - Aalborg Universitet\Master thesis\Code\Module\RMGrealtime\output\confusion_matrices.pdf"
    cm_path = _BASE_DIR / "output" / "confusion_matrices.pdf"

    export_pubfig(fig, cm_path, width=16, aspectRatio=0.75)
    print(f"✓ Confusion matrices saved to: {cm_path}")
    plt.show()