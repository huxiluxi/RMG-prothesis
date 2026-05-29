import time
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


class Classifier:
    """
    Multi-stage RF gesture classifier using standardization, PCA, and GMM.
    """
    
    def __init__(self, n_PC=30, n_components=1, verbose=False):
        """
        Initialize classifier with ML pipeline.
        
        Args:
            n_PC: Number of PCA components
            n_components: Number of Gaussian mixture components
            verbose: Persistent debug flag for timing information
        """
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_PC)
        self.lda = None
        self.gmm = GaussianMixture(n_components=n_components)
        self.fitted = False
        self.verbose = verbose
        self.label_to_gmm = {}
        self.training_data_df = None
        self.y_train = None

    def fit(self, training_data, show_data = False):
        """
        Fit classifier on training data from trainer.training_routine().
        
        Handles pandas DataFrame by extracting features and labels.
        Applies preprocessing pipeline: StandardScaler -> PCA -> GMM per class
        
        Args:
            training_data: pd.DataFrame with columns:
                - 'label': Ground truth gesture labels
                - Feature columns: 's11_mag_*', 's21_mag_*', 'feat_*'
        """
        start_time = time.time()
        
        # Extract features and labels from DataFrame
        # Drop metadata columns, keep only feature columns
        metadata_cols = ['gesture', 'iteration', 'label']
        X_train = training_data.drop(metadata_cols, axis=1).values
        y_train = training_data['label'].values
        
        if self.verbose:
            logger.debug(f"Training data shape: {X_train.shape}")
            logger.debug(f"Unique labels: {sorted(set(y_train))}")
        
        # Standardize
        t0 = time.perf_counter()
        X_scaled = self.scaler.fit_transform(X_train)
        t1 = time.perf_counter()
        
        # PCA
        t2 = time.perf_counter()
        X_pca = self.pca.fit_transform(X_scaled)
        t3 = time.perf_counter()
        
        # GMM: Train one model per gesture class
        t4 = time.perf_counter()
        unique_labels = sorted(set(y_train))
        self.label_to_gmm = {}
        for label in unique_labels:
            X_class = X_pca[y_train == label]
            gmm = GaussianMixture(n_components=self.gmm.n_components)
            gmm.fit(X_class)
            self.label_to_gmm[label] = gmm
        t5 = time.perf_counter()
        
        self.fitted = True
        
        # Timing output
        total_ms = (time.time() - start_time) * 1e3
        logger.info(f"Model fit took: {total_ms:.2f} ms")
        if self.verbose:
            logger.debug(f"Fit breakdown:")
            logger.debug(f"  StandardScaler: {(t1 - t0)*1e3:.2f} ms")
            logger.debug(f"  PCA fit:         {(t3 - t2)*1e3:.2f} ms")
            logger.debug(f"  GMM fit:         {(t5 - t4)*1e3:.2f} ms")

        if show_data:
            self._plot_training_data(X_scaled, X_pca, y_train, training_data)

    def fit_from_file(self, data_file):
        """
        Fit classifier on training data from trainer.training_routine().
        
        Handles pandas DataFrame by extracting features and labels.
        Applies preprocessing pipeline: StandardScaler -> PCA -> GMM per class
        
        Args:
            training_data: pd.DataFrame with columns:
                - 'label': Ground truth gesture labels
                - Feature columns: 's11_mag_*', 's21_mag_*', 'feat_*'
        """
        start_time = time.time()
        
        training_data = pd.read_csv(data_file)

        # Extract features and labels from DataFrame
        # Drop metadata columns, keep only feature columns
        metadata_cols = ['gesture', 'run', 'iteration', 'label']
        X_train = training_data.drop(metadata_cols, axis=1).values
        y_train = training_data['label'].values
        
        if self.verbose:
            logger.debug(f"Training data shape: {X_train.shape}")
            logger.debug(f"Unique labels: {sorted(set(y_train))}")
        
        # Standardize
        t0 = time.perf_counter()
        X_scaled = self.scaler.fit_transform(X_train)
        t1 = time.perf_counter()
        
        # PCA
        t2 = time.perf_counter()
        X_pca = self.pca.fit_transform(X_scaled)
        t3 = time.perf_counter()
        
        # GMM: Train one model per gesture class
        t4 = time.perf_counter()
        unique_labels = sorted(set(y_train))
        self.label_to_gmm = {}
        for label in unique_labels:
            X_class = X_pca[y_train == label]
            gmm = GaussianMixture(n_components=self.gmm.n_components)
            gmm.fit(X_class)
            self.label_to_gmm[label] = gmm
        t5 = time.perf_counter()
        
        self.fitted = True
        
        # Timing output
        total_ms = (time.time() - start_time) * 1e3
        logger.info(f"Model fit took: {total_ms:.2f} ms")
        if self.verbose:
            logger.debug(f"Fit breakdown:")
            logger.debug(f"  StandardScaler: {(t1 - t0)*1e3:.2f} ms")
            logger.debug(f"  PCA fit:         {(t3 - t2)*1e3:.2f} ms")
            logger.debug(f"  GMM fit:         {(t5 - t4)*1e3:.2f} ms")

    def predict(self, sample):
        """
        Predict gesture label for a sample.
        
        Args:
            sample: Feature vector (np.ndarray of shape (n_features,))
            
        Returns:
            int: Predicted label
            
        Raises:
            RuntimeError: If classifier not fitted yet
        """
        if not self.fitted:
            raise RuntimeError("Classifier not fitted yet. Call fit() first.")
        
        t0 = time.perf_counter()
        
        # Step 1: Reshape
        t1 = time.perf_counter()
        x = sample.reshape(1, -1)
        t2 = time.perf_counter()
        
        # Step 2: Scale
        t3 = time.perf_counter()
        x_scaled = self.scaler.transform(x)
        t4 = time.perf_counter()
        
        # Step 3: PCA
        t5 = time.perf_counter()
        x_pca = self.pca.transform(x_scaled)
        t6 = time.perf_counter()
        
        # Step 4: GMM - compute log probability for each class and pick highest
        t7 = time.perf_counter()
        best_label = None
        best_score = -float('inf')
        for label, gmm in self.label_to_gmm.items():
            score = gmm.score(x_pca)
            if score > best_score:
                best_score = score
                best_label = label
        t8 = time.perf_counter()
        
        if self.verbose:
            logger.debug(f"Predict breakdown:")
            logger.debug(f"  Reshape:        {(t2 - t1)*1e6:.2f} µs")
            logger.debug(f"  StandardScaler: {(t4 - t3)*1e6:.2f} µs")
            logger.debug(f"  PCA transform:  {(t6 - t5)*1e6:.2f} µs")
            logger.debug(f"  GMM predict:    {(t8 - t7)*1e6:.2f} µs")
            logger.debug(f"  Total predict:  {(t8 - t0)*1e6:.2f} µs")
        
        return best_label

    def _plot_training_data(self, X_scaled, X_pca, y_train, training_data):
        """
        Plot training data using PCA and LDA dimensionality reduction.
        
        Args:
            X_scaled: Standardized feature matrix
            X_pca: PCA-transformed feature matrix (2D)
            y_train: Training labels
            training_data: Original DataFrame with metadata
        """
        # Prepare LDA if we have enough components
        try:
            n_classes = len(set(y_train))
            lda = LDA(n_components=min(2, n_classes - 1))
            X_lda = lda.fit_transform(X_scaled, y_train)
        except Exception as e:
            logger.warning(f"LDA fit failed: {e}. Skipping LDA plot.")
            X_lda = None
        
        # Get gesture names and markers
        gesture_names = training_data['gesture'].unique()
        markers = ['o', 's', '^', 'v', '<', '>', 'D', 'P', 'X', '*', '+', 'x']
        cmap = plt.get_cmap('tab20', len(gesture_names))
        
        # Create figures
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        fig2, ax2 = plt.subplots(figsize=(10, 8)) if X_lda is not None else (None, None)
        
        plotted_gestures = set()
        total_plotted = 0
        
        # Plot each gesture
        for i, gesture in enumerate(gesture_names):
            gesture_indices = (training_data['gesture'] == gesture).values
            
            # Plot PCA
            ax1.scatter(X_pca[gesture_indices, 0],
                       X_pca[gesture_indices, 1],
                       color=cmap(i),
                       marker='o',
                       s=30,
                       alpha=0.6,
                       label=gesture)
            
            # Plot LDA if available
            if X_lda is not None and ax2 is not None:
                ax2.scatter(X_lda[gesture_indices, 0],
                           X_lda[gesture_indices, 1],
                           color=cmap(i),
                           marker='o',
                           s=30,
                           alpha=0.6,
                           label=gesture)
            
            total_plotted += np.sum(gesture_indices)
        
        # Configure PCA plot
        ax1.set_title("PCA Dimensionality Reduction (Training Data)")
        ax1.set_xlabel("PC₁")
        ax1.set_ylabel("PC₂")
        ax1.legend(title="Gestures", fontsize='small', loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Configure LDA plot if available
        if X_lda is not None and ax2 is not None:
            ax2.set_title("LDA Dimensionality Reduction (Training Data)")
            ax2.set_xlabel("LDA₁")
            ax2.set_ylabel("LDA₂")
            ax2.legend(title="Gestures", fontsize='small', loc='best')
            ax2.grid(True, alpha=0.3)
        
        logger.info(f"Plotted {total_plotted} training samples across {len(gesture_names)} gestures")
        plt.tight_layout()
        plt.show()