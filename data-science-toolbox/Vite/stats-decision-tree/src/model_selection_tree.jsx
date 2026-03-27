import { useState } from "react";

// ─────────────────────────────────────────────────────────────────
// DECISION TREE DATA
// Each node: { id, question, options: [{label, next}] }
// Leaf nodes: { id, models: [...], notes, tip }
// ─────────────────────────────────────────────────────────────────
const TREE = {
  q_task: {
    question: "What type of prediction task are you solving?",
    subtitle: "Start here — everything flows from your task type.",
    options: [
      { label: "Predict a category / class", icon: "⬡", next: "q_target_type" },
      { label: "Predict a number", icon: "∿", next: "q_regression_size" },
      { label: "Find hidden structure / groups", icon: "◎", next: "q_cluster_labels" },
      { label: "Predict sequences or time", icon: "⟶", next: "q_time_size" },
      { label: "Generate or transform content", icon: "✦", next: "leaf_generative" },
    ],
  },

  // ── CLASSIFICATION BRANCH ──────────────────────────────────────
  q_target_type: {
    question: "How many classes does your target have?",
    options: [
      { label: "Two classes (binary)", icon: "◐", next: "q_imbalance" },
      { label: "Three or more classes", icon: "◉", next: "q_multiclass_size" },
    ],
  },
  q_imbalance: {
    question: "Is your dataset class-imbalanced?",
    subtitle: "Imbalanced = minority class < ~15% of total.",
    options: [
      { label: "Yes — minority class is rare", icon: "⚠", next: "q_imbalance_size" },
      { label: "No — classes are roughly balanced", icon: "✓", next: "q_binary_size" },
    ],
  },
  q_imbalance_size: {
    question: "How large is your dataset?",
    options: [
      { label: "Small (< 10k rows)", icon: "·", next: "leaf_binary_imb_small" },
      { label: "Medium (10k – 500k rows)", icon: "··", next: "leaf_binary_imb_med" },
      { label: "Large (> 500k rows)", icon: "···", next: "leaf_binary_imb_large" },
    ],
  },
  q_binary_size: {
    question: "How large is your dataset?",
    options: [
      { label: "Small (< 10k rows)", icon: "·", next: "leaf_binary_bal_small" },
      { label: "Medium (10k – 500k rows)", icon: "··", next: "leaf_binary_bal_med" },
      { label: "Large (> 500k rows)", icon: "···", next: "leaf_binary_bal_large" },
    ],
  },
  q_multiclass_size: {
    question: "How large is your dataset?",
    options: [
      { label: "Small (< 10k rows)", icon: "·", next: "leaf_multi_small" },
      { label: "Medium / Large (> 10k rows)", icon: "··", next: "leaf_multi_large" },
    ],
  },

  // ── REGRESSION BRANCH ─────────────────────────────────────────
  q_regression_size: {
    question: "What is the nature of your target variable?",
    options: [
      { label: "Continuous (any real number)", icon: "∼", next: "q_reg_features" },
      { label: "Count (non-negative integers)", icon: "ℕ", next: "leaf_count_reg" },
      { label: "Bounded / Proportion (0–1)", icon: "%", next: "leaf_bounded_reg" },
    ],
  },
  q_reg_features: {
    question: "What is your feature space like?",
    options: [
      { label: "Mostly linear relationships expected", icon: "⟋", next: "q_reg_linear_size" },
      { label: "Complex / non-linear relationships", icon: "⌇", next: "q_reg_nonlinear_size" },
      { label: "Lots of features (> 50 columns)", icon: "≡", next: "leaf_reg_highdim" },
    ],
  },
  q_reg_linear_size: {
    question: "Dataset size?",
    options: [
      { label: "Small–Medium (< 100k rows)", icon: "·", next: "leaf_reg_linear_small" },
      { label: "Large (> 100k rows)", icon: "···", next: "leaf_reg_linear_large" },
    ],
  },
  q_reg_nonlinear_size: {
    question: "Dataset size?",
    options: [
      { label: "Small (< 10k rows)", icon: "·", next: "leaf_reg_nl_small" },
      { label: "Medium–Large (> 10k rows)", icon: "··", next: "leaf_reg_nl_large" },
    ],
  },

  // ── CLUSTERING BRANCH ─────────────────────────────────────────
  q_cluster_labels: {
    question: "Do you have any labeled data?",
    options: [
      { label: "No labels at all — fully unsupervised", icon: "?", next: "q_cluster_shape" },
      { label: "Some labels — semi-supervised", icon: "½", next: "leaf_semi_supervised" },
    ],
  },
  q_cluster_shape: {
    question: "Do you know how many clusters you want?",
    options: [
      { label: "Yes — I know the number of clusters", icon: "✓", next: "q_cluster_known_shape" },
      { label: "No — let the algorithm decide", icon: "?", next: "leaf_cluster_unknown" },
    ],
  },
  q_cluster_known_shape: {
    question: "What shape do you expect your clusters to be?",
    options: [
      { label: "Roughly spherical / convex", icon: "●", next: "leaf_cluster_spherical" },
      { label: "Irregular / elongated / arbitrary", icon: "⌘", next: "leaf_cluster_irregular" },
    ],
  },

  // ── TIME SERIES BRANCH ────────────────────────────────────────
  q_time_size: {
    question: "What are you trying to do with time data?",
    options: [
      { label: "Forecast future values", icon: "→", next: "q_ts_freq" },
      { label: "Classify time windows / sequences", icon: "⬡", next: "leaf_ts_classify" },
      { label: "Detect anomalies over time", icon: "⚡", next: "leaf_ts_anomaly" },
    ],
  },
  q_ts_freq: {
    question: "What is your data frequency and volume?",
    options: [
      { label: "Low frequency (daily/weekly/monthly), few series", icon: "📅", next: "leaf_ts_classic" },
      { label: "High frequency or many series", icon: "⚡", next: "leaf_ts_ml" },
    ],
  },

  // ─────────────────────────────────────────────────────────────
  // LEAF NODES — MODEL RECOMMENDATIONS
  // ─────────────────────────────────────────────────────────────

  leaf_binary_imb_small: {
    models: ["Logistic Regression (class_weight='balanced')", "Random Forest (balanced)", "XGBoost (scale_pos_weight)"],
    primary: "Logistic Regression + SMOTE",
    notes: "Small + imbalanced is tough. Use SMOTE oversampling on training set only. Evaluate with AUC-ROC and Precision-Recall, not accuracy.",
    tip: "Try threshold tuning on your probability output. Default 0.5 is rarely optimal for imbalanced data.",
    metrics: ["AUC-ROC", "F1 (minority)", "Precision-Recall AUC"],
  },
  leaf_binary_imb_med: {
    models: ["XGBoost (scale_pos_weight)", "LightGBM (is_unbalance=True)", "Random Forest (class_weight='balanced')", "Logistic Regression (baseline)"],
    primary: "LightGBM or XGBoost",
    notes: "Gradient boosting handles imbalance well with built-in weight parameters. Always keep Logistic Regression as interpretable baseline.",
    tip: "Set scale_pos_weight = n_negatives / n_positives in XGBoost. Use RandomizedSearchCV for tuning.",
    metrics: ["AUC-ROC", "KS Statistic", "Gini Coefficient", "Top-decile capture"],
  },
  leaf_binary_imb_large: {
    models: ["LightGBM (fastest at scale)", "XGBoost", "Logistic Regression (baseline)"],
    primary: "LightGBM",
    notes: "At large scale, LightGBM's histogram-based algorithm is significantly faster than XGBoost. Avoid RandomForest at this scale — memory intensive.",
    tip: "Use early stopping with a validation set to avoid overfitting without expensive grid search.",
    metrics: ["AUC-ROC", "Calibration (Brier Score)", "Throughput / inference speed"],
  },
  leaf_binary_bal_small: {
    models: ["Logistic Regression", "Decision Tree", "SVM (RBF kernel)", "Random Forest"],
    primary: "Logistic Regression → Random Forest",
    notes: "Start simple. Logistic Regression gives you a strong, interpretable baseline. Only add complexity if it clearly improves validation AUC.",
    tip: "With small data, cross-validation is essential. Use 5-fold stratified CV, not a single train/val split.",
    metrics: ["AUC-ROC", "F1 Score", "CV stability (std across folds)"],
  },
  leaf_binary_bal_med: {
    models: ["Logistic Regression (baseline)", "Random Forest", "Gradient Boosting", "LightGBM", "XGBoost"],
    primary: "LightGBM or XGBoost",
    notes: "This is the classic DS sweet spot. Run all five, compare on validation AUC, tune the top 2.",
    tip: "This is exactly the workflow in your Loan Default toolkit — notebooks 05 and 06 are built for this.",
    metrics: ["AUC-ROC", "Precision-Recall AUC", "Confusion matrix at optimal threshold"],
  },
  leaf_binary_bal_large: {
    models: ["LightGBM", "XGBoost", "Logistic Regression (baseline)"],
    primary: "LightGBM",
    notes: "LightGBM is the workhorse for large-scale classification. Scales to hundreds of millions of rows efficiently.",
    tip: "Consider mini-batch training or Dask for very large datasets that don't fit in RAM.",
    metrics: ["AUC-ROC", "Inference latency", "Model size"],
  },
  leaf_multi_small: {
    models: ["Logistic Regression (multinomial)", "Decision Tree", "Random Forest", "KNN"],
    primary: "Logistic Regression → Random Forest",
    notes: "Multi-class with small data: keep it simple. Beware of per-class imbalance — stratify your splits.",
    tip: "Check confusion matrix for each class. Some classes may consistently confuse the model.",
    metrics: ["Macro F1", "Per-class precision & recall", "Confusion matrix"],
  },
  leaf_multi_large: {
    models: ["LightGBM (multi:softmax)", "XGBoost (multi:softmax)", "Random Forest", "Logistic Regression (baseline)"],
    primary: "LightGBM",
    notes: "Gradient boosting handles multi-class natively. Set objective='multiclass' and num_class accordingly.",
    tip: "For 10+ classes, consider whether the problem can be restructured as One-vs-Rest or hierarchical classification.",
    metrics: ["Macro F1", "Weighted F1", "Top-k accuracy if applicable"],
  },
  leaf_reg_linear_small: {
    models: ["Linear Regression", "Ridge Regression", "Lasso Regression", "ElasticNet"],
    primary: "Ridge → ElasticNet",
    notes: "Ridge adds L2 regularization (prevents overfitting). Lasso does L1 (forces feature selection). ElasticNet combines both.",
    tip: "Always check residual plots. Non-random patterns mean your linear assumption is wrong.",
    metrics: ["RMSE", "MAE", "R²", "Residual normality"],
  },
  leaf_reg_linear_large: {
    models: ["Ridge Regression", "SGD Regressor (mini-batch)", "Linear Regression"],
    primary: "SGD Regressor",
    notes: "SGD (Stochastic Gradient Descent) allows mini-batch training — essential when data doesn't fit in RAM.",
    tip: "Standardize all features before fitting. SGD is very sensitive to feature scale.",
    metrics: ["RMSE", "MAE", "R²"],
  },
  leaf_reg_nl_small: {
    models: ["Decision Tree Regressor", "Random Forest Regressor", "SVR (RBF kernel)", "Polynomial Regression"],
    primary: "Random Forest Regressor",
    notes: "Random Forest is robust to outliers and doesn't require feature scaling. SVR is powerful but slow on large data.",
    tip: "With small data and high complexity, overfitting is the #1 risk. Use CV, not a single validation split.",
    metrics: ["RMSE", "MAE", "R²", "CV RMSE std"],
  },
  leaf_reg_nl_large: {
    models: ["LightGBM Regressor", "XGBoost Regressor", "Gradient Boosting Regressor", "Neural Network (MLP)"],
    primary: "LightGBM Regressor",
    notes: "Gradient boosting dominates tabular regression. LightGBM is fastest. Neural nets only win when you have 1M+ rows with complex interactions.",
    tip: "Set objective='regression' (LightGBM) or 'reg:squarederror' (XGBoost). Tune learning_rate and n_estimators together.",
    metrics: ["RMSE", "MAE", "MAPE (if no zeros)", "R²"],
  },
  leaf_reg_highdim: {
    models: ["Lasso Regression", "ElasticNet", "Ridge Regression", "Random Forest (with feature importance)", "PCA + Regression"],
    primary: "Lasso or ElasticNet",
    notes: "High-dimensional data needs regularization or dimensionality reduction. Lasso zeroes out irrelevant features automatically.",
    tip: "Run Lasso first — zero coefficients tell you which features to drop. Then try tree models on the reduced set.",
    metrics: ["RMSE", "MAE", "R²", "# non-zero coefficients"],
  },
  leaf_count_reg: {
    models: ["Poisson Regression", "Negative Binomial Regression", "XGBoost (count:poisson)", "Random Forest Regressor"],
    primary: "Poisson Regression → XGBoost",
    notes: "Count data (events, transactions, visits) violates linear regression assumptions. Use count-specific models.",
    tip: "Check for overdispersion: if Var(y) >> Mean(y), use Negative Binomial instead of Poisson.",
    metrics: ["Mean Poisson Deviance", "MAE", "RMSE"],
  },
  leaf_bounded_reg: {
    models: ["Beta Regression", "Logistic Regression (as probability calibrator)", "Gradient Boosting + sigmoid output"],
    primary: "Beta Regression",
    notes: "Proportions (0–1) need a model that respects the boundary. Beta Regression is purpose-built for this.",
    tip: "Values of exactly 0 or 1 cause issues for Beta Regression — consider a zero-inflated model or small epsilon adjustment.",
    metrics: ["MAE", "Brier Score", "Calibration curve"],
  },
  leaf_cluster_spherical: {
    models: ["K-Means", "K-Means++", "Mini-Batch K-Means (large data)", "Gaussian Mixture Models (GMM)"],
    primary: "K-Means++",
    notes: "K-Means is fast and interpretable. K-Means++ uses smarter initialization to avoid local optima. GMM gives soft cluster assignments (probabilities).",
    tip: "Validate k using Elbow Method (inertia) AND Silhouette Score — they sometimes disagree. Always plot both.",
    metrics: ["Silhouette Score", "Inertia (Elbow)", "Davies-Bouldin Index"],
  },
  leaf_cluster_irregular: {
    models: ["DBSCAN", "HDBSCAN", "Agglomerative Clustering", "Spectral Clustering"],
    primary: "HDBSCAN",
    notes: "For non-spherical clusters, density-based methods win. DBSCAN finds arbitrary shapes and marks noise points. HDBSCAN is more robust to parameter choice.",
    tip: "DBSCAN requires tuning eps and min_samples. Use a k-distance plot to find eps. HDBSCAN reduces this to just min_cluster_size.",
    metrics: ["Silhouette Score", "DBCV (density-based)", "Visual inspection"],
  },
  leaf_cluster_unknown: {
    models: ["HDBSCAN", "DBSCAN", "Gaussian Mixture Model (BIC to pick k)", "Agglomerative + Dendrogram"],
    primary: "HDBSCAN",
    notes: "When you don't know k, use methods that determine structure automatically. HDBSCAN is usually the best starting point.",
    tip: "With GMM, use BIC or AIC across k=2..20 to find the best number of components. Plot the curve — the elbow is your k.",
    metrics: ["Silhouette Score", "BIC/AIC (for GMM)", "Stability across runs"],
  },
  leaf_semi_supervised: {
    models: ["Label Propagation", "Label Spreading", "Self-Training Classifier (sklearn)", "Co-Training"],
    primary: "Label Spreading",
    notes: "Semi-supervised learning uses the graph structure of unlabeled data to propagate labels. Works best when labeled and unlabeled data share the same distribution.",
    tip: "If you have even 5–10% labeled data, consider pseudo-labeling: train on labeled, predict unlabeled, add high-confidence predictions, retrain.",
    metrics: ["Accuracy on labeled held-out set", "Consistency of predicted labels"],
  },
  leaf_ts_classic: {
    models: ["ARIMA / SARIMA", "Exponential Smoothing (ETS)", "Prophet (Facebook)", "Theta Model"],
    primary: "Prophet → ARIMA",
    notes: "For daily/weekly/monthly business time series, Prophet handles seasonality and holidays automatically. ARIMA is the statistical gold standard.",
    tip: "Always check for stationarity (ADF test) before ARIMA. Difference your series if it trends. Use SARIMA for strong seasonal patterns.",
    metrics: ["MAE", "RMSE", "MAPE", "Coverage (prediction intervals)"],
  },
  leaf_ts_ml: {
    models: ["LightGBM with lag features", "XGBoost with lag features", "LSTM / GRU (deep learning)", "N-BEATS / Temporal Fusion Transformer"],
    primary: "LightGBM with engineered lag features",
    notes: "Transform the time series into a supervised problem using lag features, rolling statistics, and calendar features. Gradient boosting then outperforms classical methods at scale.",
    tip: "Feature engineering is everything here: lag_1, lag_7, lag_30, rolling_mean_7, rolling_std_14, day_of_week, month, is_holiday.",
    metrics: ["MAE", "RMSE", "SMAPE", "Directional accuracy"],
  },
  leaf_ts_classify: {
    models: ["Random Forest (on feature windows)", "LightGBM", "1D CNN", "LSTM", "Rocket / MiniRocket (state-of-the-art)"],
    primary: "Rocket / MiniRocket",
    notes: "For classifying time windows (e.g. ECG, sensor readings), feature-based methods on sliding windows work well. Rocket uses random convolutional kernels and is extremely fast.",
    tip: "Extract features from windows: mean, std, min, max, autocorrelation, FFT components. Then apply a standard classifier.",
    metrics: ["Accuracy", "F1 per class", "AUC-ROC"],
  },
  leaf_ts_anomaly: {
    models: ["Isolation Forest", "LSTM Autoencoder", "STL Decomposition + thresholding", "Prophet residual analysis"],
    primary: "Isolation Forest → LSTM Autoencoder",
    notes: "Isolation Forest is fast and works well for point anomalies. LSTM Autoencoder catches contextual anomalies (unusual patterns, not just unusual values).",
    tip: "Define 'anomaly' carefully before modeling. Point anomaly (single outlier) vs contextual anomaly (unusual given context) vs collective anomaly (unusual subsequence) need different approaches.",
    metrics: ["Precision@k", "Recall", "F1 (if labeled)", "Visual inspection"],
  },
  leaf_generative: {
    models: ["Fine-tuned LLM (text)", "Stable Diffusion (images)", "VAE / GAN (structured data synthesis)", "SMOTE (tabular oversampling)"],
    primary: "Depends on modality",
    notes: "Generative tasks are highly modality-dependent. For tabular data synthesis, SDV (Synthetic Data Vault) or CTGAN are purpose-built. For text, fine-tuning an LLM via the Anthropic API is the fastest path.",
    tip: "For tabular data augmentation specifically: CTGAN and TVAE from the SDV library outperform basic SMOTE for complex distributions.",
    metrics: ["Fidelity (statistical similarity)", "Utility (downstream ML performance)", "Privacy (distance to real records)"],
  },
};

// ─────────────────────────────────────────────────────────────────
// COLOUR & STYLE CONSTANTS
// ─────────────────────────────────────────────────────────────────
const PALETTE = {
  bg: "#0d0f14",
  surface: "#13161e",
  card: "#191d28",
  border: "#252a38",
  accent: "#4fffb0",
  accentDim: "#1a3d2e",
  gold: "#ffd166",
  goldDim: "#3a3015",
  text: "#e8eaf2",
  muted: "#6b7280",
  crumb: "#a0a8bf",
};

// ─────────────────────────────────────────────────────────────────
// COMPONENT
// ─────────────────────────────────────────────────────────────────
export default function ModelDecisionTree() {
  const [path, setPath] = useState(["q_task"]);
  const currentId = path[path.length - 1];
  const currentNode = TREE[currentId];
  const isLeaf = !!currentNode?.models;

  const choose = (nextId) => setPath([...path, nextId]);
  const goBack = () => setPath(path.slice(0, -1));
  const restart = () => setPath(["q_task"]);
  const goTo = (idx) => setPath(path.slice(0, idx + 1));

  const breadcrumbs = path.map((id) => {
    const node = TREE[id];
    if (!node) return null;
    return node.question || "Result";
  });

  return (
    <div style={{
      minHeight: "100vh",
      background: PALETTE.bg,
      color: PALETTE.text,
      fontFamily: "'IBM Plex Mono', 'Courier New', monospace",
      padding: "0",
    }}>
      {/* ── HEADER ─────────────────────────────────────────── */}
      <div style={{
        borderBottom: `1px solid ${PALETTE.border}`,
        padding: "24px 32px 20px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        background: PALETTE.surface,
      }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 4 }}>
            <span style={{
              color: PALETTE.accent,
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.2em",
              textTransform: "uppercase",
              border: `1px solid ${PALETTE.accent}`,
              padding: "2px 8px",
            }}>DS Toolkit</span>
          </div>
          <h1 style={{
            margin: 0,
            fontSize: 22,
            fontWeight: 700,
            letterSpacing: "-0.01em",
            color: PALETTE.text,
            fontFamily: "'IBM Plex Mono', monospace",
          }}>Model Selection Navigator</h1>
        </div>
        <button
          onClick={restart}
          style={{
            background: "transparent",
            border: `1px solid ${PALETTE.border}`,
            color: PALETTE.muted,
            padding: "8px 16px",
            cursor: "pointer",
            fontSize: 12,
            letterSpacing: "0.1em",
            fontFamily: "inherit",
          }}
        >
          ↺ RESTART
        </button>
      </div>

      {/* ── BREADCRUMB ─────────────────────────────────────── */}
      <div style={{
        padding: "14px 32px",
        borderBottom: `1px solid ${PALETTE.border}`,
        display: "flex",
        alignItems: "center",
        flexWrap: "wrap",
        gap: 4,
        fontSize: 11,
        color: PALETTE.muted,
        background: PALETTE.surface,
      }}>
        {path.map((id, idx) => {
          const node = TREE[id];
          const label = idx === 0 ? "Task Type" :
            node?.models ? "Result" :
            node?.question?.split(" ").slice(0, 5).join(" ") + "…";
          const isLast = idx === path.length - 1;
          return (
            <span key={id} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              {idx > 0 && <span style={{ color: PALETTE.border }}>›</span>}
              <span
                onClick={() => !isLast && goTo(idx)}
                style={{
                  color: isLast ? PALETTE.accent : PALETTE.crumb,
                  cursor: isLast ? "default" : "pointer",
                  textDecoration: isLast ? "none" : "underline",
                  textUnderlineOffset: 3,
                }}
              >
                {label}
              </span>
            </span>
          );
        })}
      </div>

      {/* ── MAIN CONTENT ───────────────────────────────────── */}
      <div style={{ padding: "40px 32px", maxWidth: 820, margin: "0 auto" }}>

        {/* ── QUESTION NODE ─────────────────────────────────── */}
        {!isLeaf && (
          <>
            <div style={{ marginBottom: 32 }}>
              <div style={{
                fontSize: 11,
                color: PALETTE.muted,
                letterSpacing: "0.15em",
                textTransform: "uppercase",
                marginBottom: 12,
              }}>
                STEP {path.length} OF {path.length}
              </div>
              <h2 style={{
                margin: "0 0 8px",
                fontSize: 26,
                fontWeight: 700,
                lineHeight: 1.3,
                color: PALETTE.text,
                letterSpacing: "-0.02em",
                fontFamily: "inherit",
              }}>
                {currentNode.question}
              </h2>
              {currentNode.subtitle && (
                <p style={{ margin: 0, color: PALETTE.muted, fontSize: 14, lineHeight: 1.6 }}>
                  {currentNode.subtitle}
                </p>
              )}
            </div>

            {/* Options */}
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {currentNode.options.map((opt, i) => (
                <button
                  key={i}
                  onClick={() => choose(opt.next)}
                  style={{
                    background: PALETTE.card,
                    border: `1px solid ${PALETTE.border}`,
                    color: PALETTE.text,
                    padding: "18px 24px",
                    cursor: "pointer",
                    textAlign: "left",
                    fontFamily: "inherit",
                    fontSize: 15,
                    display: "flex",
                    alignItems: "center",
                    gap: 16,
                    transition: "all 0.15s",
                    position: "relative",
                    overflow: "hidden",
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.borderColor = PALETTE.accent;
                    e.currentTarget.style.background = "#1a1f2e";
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.borderColor = PALETTE.border;
                    e.currentTarget.style.background = PALETTE.card;
                  }}
                >
                  <span style={{
                    fontSize: 20,
                    color: PALETTE.accent,
                    minWidth: 28,
                    textAlign: "center",
                    opacity: 0.9,
                  }}>{opt.icon}</span>
                  <span style={{ flex: 1 }}>{opt.label}</span>
                  <span style={{ color: PALETTE.muted, fontSize: 18 }}>›</span>
                </button>
              ))}
            </div>

            {path.length > 1 && (
              <button
                onClick={goBack}
                style={{
                  marginTop: 24,
                  background: "transparent",
                  border: "none",
                  color: PALETTE.muted,
                  cursor: "pointer",
                  fontFamily: "inherit",
                  fontSize: 13,
                  padding: 0,
                  letterSpacing: "0.05em",
                }}
              >
                ← Back
              </button>
            )}
          </>
        )}

        {/* ── LEAF NODE — RESULTS ───────────────────────────── */}
        {isLeaf && (
          <>
            {/* Header */}
            <div style={{ marginBottom: 28 }}>
              <div style={{
                fontSize: 11,
                color: PALETTE.accent,
                letterSpacing: "0.2em",
                textTransform: "uppercase",
                marginBottom: 10,
              }}>
                ✦ Recommendation Ready
              </div>
              <h2 style={{
                margin: 0,
                fontSize: 24,
                fontWeight: 700,
                color: PALETTE.text,
                fontFamily: "inherit",
                letterSpacing: "-0.02em",
              }}>
                Your Model Stack
              </h2>
            </div>

            {/* Primary Recommendation */}
            <div style={{
              background: PALETTE.accentDim,
              border: `1px solid ${PALETTE.accent}`,
              padding: "22px 26px",
              marginBottom: 16,
            }}>
              <div style={{
                fontSize: 10,
                letterSpacing: "0.2em",
                color: PALETTE.accent,
                textTransform: "uppercase",
                marginBottom: 8,
                fontWeight: 700,
              }}>
                ★ Start With
              </div>
              <div style={{
                fontSize: 19,
                fontWeight: 700,
                color: PALETTE.accent,
                letterSpacing: "-0.01em",
              }}>
                {currentNode.primary}
              </div>
            </div>

            {/* All Models */}
            <div style={{
              background: PALETTE.card,
              border: `1px solid ${PALETTE.border}`,
              padding: "22px 26px",
              marginBottom: 16,
            }}>
              <div style={{
                fontSize: 10,
                letterSpacing: "0.2em",
                color: PALETTE.muted,
                textTransform: "uppercase",
                marginBottom: 14,
                fontWeight: 700,
              }}>
                Full Candidate List (run in order)
              </div>
              {currentNode.models.map((m, i) => (
                <div key={i} style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 12,
                  padding: "9px 0",
                  borderTop: i === 0 ? "none" : `1px solid ${PALETTE.border}`,
                }}>
                  <span style={{
                    color: PALETTE.muted,
                    fontSize: 11,
                    minWidth: 20,
                    textAlign: "right",
                  }}>
                    {String(i + 1).padStart(2, "0")}
                  </span>
                  <span style={{ fontSize: 14, color: PALETTE.text }}>{m}</span>
                </div>
              ))}
            </div>

            {/* Notes */}
            <div style={{
              background: PALETTE.card,
              border: `1px solid ${PALETTE.border}`,
              padding: "22px 26px",
              marginBottom: 16,
            }}>
              <div style={{
                fontSize: 10,
                letterSpacing: "0.2em",
                color: PALETTE.muted,
                textTransform: "uppercase",
                marginBottom: 12,
                fontWeight: 700,
              }}>
                Context & Rationale
              </div>
              <p style={{
                margin: 0,
                fontSize: 14,
                lineHeight: 1.75,
                color: PALETTE.crumb,
              }}>
                {currentNode.notes}
              </p>
            </div>

            {/* Pro Tip */}
            <div style={{
              background: PALETTE.goldDim,
              border: `1px solid ${PALETTE.gold}`,
              padding: "18px 24px",
              marginBottom: 16,
              display: "flex",
              gap: 14,
              alignItems: "flex-start",
            }}>
              <span style={{ fontSize: 18, color: PALETTE.gold, marginTop: 1 }}>⚡</span>
              <div>
                <div style={{
                  fontSize: 10,
                  letterSpacing: "0.2em",
                  color: PALETTE.gold,
                  textTransform: "uppercase",
                  marginBottom: 6,
                  fontWeight: 700,
                }}>
                  Pro Tip
                </div>
                <p style={{
                  margin: 0,
                  fontSize: 13,
                  lineHeight: 1.7,
                  color: "#d4b96a",
                }}>
                  {currentNode.tip}
                </p>
              </div>
            </div>

            {/* Evaluation Metrics */}
            <div style={{
              background: PALETTE.card,
              border: `1px solid ${PALETTE.border}`,
              padding: "20px 26px",
              marginBottom: 28,
            }}>
              <div style={{
                fontSize: 10,
                letterSpacing: "0.2em",
                color: PALETTE.muted,
                textTransform: "uppercase",
                marginBottom: 12,
                fontWeight: 700,
              }}>
                Evaluate With
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {currentNode.metrics.map((m, i) => (
                  <span key={i} style={{
                    background: PALETTE.surface,
                    border: `1px solid ${PALETTE.border}`,
                    padding: "5px 12px",
                    fontSize: 12,
                    color: PALETTE.crumb,
                    letterSpacing: "0.03em",
                  }}>
                    {m}
                  </span>
                ))}
              </div>
            </div>

            {/* Actions */}
            <div style={{ display: "flex", gap: 12 }}>
              <button
                onClick={goBack}
                style={{
                  background: "transparent",
                  border: `1px solid ${PALETTE.border}`,
                  color: PALETTE.muted,
                  padding: "12px 20px",
                  cursor: "pointer",
                  fontFamily: "inherit",
                  fontSize: 13,
                  letterSpacing: "0.05em",
                }}
              >
                ← Back
              </button>
              <button
                onClick={restart}
                style={{
                  background: PALETTE.accentDim,
                  border: `1px solid ${PALETTE.accent}`,
                  color: PALETTE.accent,
                  padding: "12px 24px",
                  cursor: "pointer",
                  fontFamily: "inherit",
                  fontSize: 13,
                  letterSpacing: "0.1em",
                  fontWeight: 700,
                }}
              >
                ↺ START NEW QUERY
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
