//! Classification metrics for evaluating CNC results.
//!
//! This module provides standard classification metrics:
//! - Accuracy, Recall, Precision, F1-score
//! - Matthews Correlation Coefficient (MCC)
//! - ROC Area Under Curve (AUC-ROC)
//! - Precision-Recall Curve Area (PRC Area)

use std::collections::HashMap;
use crate::cnc::{CncResult, NominalDataset};

/// Confusion matrix for binary and multi-class classification
#[derive(Debug, Clone)]
pub struct ConfusionMatrix {
    /// Map of (actual_class, predicted_class) -> count
    pub matrix: HashMap<(String, String), usize>,
    /// All unique classes
    pub classes: Vec<String>,
    /// Total number of samples
    pub total: usize,
}

impl ConfusionMatrix {
    /// Create a new confusion matrix from actual and predicted labels
    pub fn new(actual: &[String], predicted: &[String]) -> Self {
        assert_eq!(actual.len(), predicted.len(), "actual and predicted must have same length");

        let mut matrix = HashMap::new();
        let mut classes_set = std::collections::HashSet::new();

        for (a, p) in actual.iter().zip(predicted.iter()) {
            classes_set.insert(a.clone());
            classes_set.insert(p.clone());
            *matrix.entry((a.clone(), p.clone())).or_insert(0) += 1;
        }

        let mut classes: Vec<String> = classes_set.into_iter().collect();
        classes.sort();

        ConfusionMatrix {
            matrix,
            classes,
            total: actual.len(),
        }
    }

    /// Get count for a specific (actual, predicted) pair
    pub fn get(&self, actual: &str, predicted: &str) -> usize {
        *self.matrix.get(&(actual.to_string(), predicted.to_string())).unwrap_or(&0)
    }

    /// Calculate True Positives for a specific class
    pub fn true_positives(&self, class: &str) -> usize {
        self.get(class, class)
    }

    /// Calculate False Positives for a specific class (predicted as class but wasn't)
    pub fn false_positives(&self, class: &str) -> usize {
        self.classes.iter()
            .filter(|c| *c != class)
            .map(|c| self.get(c, class))
            .sum()
    }

    /// Calculate False Negatives for a specific class (was class but predicted as something else)
    pub fn false_negatives(&self, class: &str) -> usize {
        self.classes.iter()
            .filter(|c| *c != class)
            .map(|c| self.get(class, c))
            .sum()
    }

    /// Calculate True Negatives for a specific class
    pub fn true_negatives(&self, class: &str) -> usize {
        self.classes.iter()
            .filter(|actual| *actual != class)
            .flat_map(|actual| {
                self.classes.iter()
                    .filter(|pred| *pred != class)
                    .map(move |pred| self.get(actual, pred))
            })
            .sum()
    }
}

/// Classification metrics results
#[derive(Debug, Clone)]
pub struct ClassificationMetrics {
    pub accuracy: f64,
    pub macro_precision: f64,
    pub macro_recall: f64,
    pub macro_f1: f64,
    pub mcc: f64,
    pub roc_auc: f64,
    pub prc_auc: f64,
    /// Per-class metrics
    pub per_class: HashMap<String, PerClassMetrics>,
    /// Number of objects covered by concepts
    pub coverage: usize,
    /// Total number of objects
    pub total: usize,
}

/// Per-class metrics
#[derive(Debug, Clone)]
pub struct PerClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize, // Number of actual instances of this class
}

impl std::fmt::Display for ClassificationMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Classification Metrics:")?;
        writeln!(f, "  Coverage: {}/{} ({:.1}%)",
            self.coverage, self.total,
            (self.coverage as f64 / self.total as f64) * 100.0)?;
        writeln!(f, "  Accuracy:  {:.4}", self.accuracy)?;
        writeln!(f, "  Recall:    {:.4} (macro)", self.macro_recall)?;
        writeln!(f, "  Precision: {:.4} (macro)", self.macro_precision)?;
        writeln!(f, "  F1-score:  {:.4} (macro)", self.macro_f1)?;
        writeln!(f, "  MCC:       {:.4}", self.mcc)?;
        writeln!(f, "  ROC AUC:   {:.4}", self.roc_auc)?;
        writeln!(f, "  PRC AUC:   {:.4}", self.prc_auc)?;

        writeln!(f, "\n  Per-class metrics:")?;
        let mut classes: Vec<_> = self.per_class.keys().collect();
        classes.sort();
        for class in classes {
            let m = &self.per_class[class];
            writeln!(f, "    {}: P={:.3} R={:.3} F1={:.3} (support={})",
                class, m.precision, m.recall, m.f1, m.support)?;
        }
        Ok(())
    }
}

/// Prediction with confidence score for ROC/PRC calculations
#[derive(Debug, Clone)]
pub struct Prediction {
    pub object_idx: usize,
    pub predicted_class: String,
    pub confidence: f64, // Confidence score (0-1)
}

/// Evaluate CNC results as a classifier
///
/// Each object covered by a concept is assigned the majority class of that concept.
/// The confidence is the proportion of the majority class in the concept's extent.
/// Objects covered by multiple concepts use the concept with highest confidence.
/// Objects NOT covered by any concept are assigned the dataset's majority class
/// with a low confidence, ensuring realistic evaluation metrics.
pub fn evaluate_cnc(dataset: &NominalDataset, result: &CncResult) -> ClassificationMetrics {
    let n_objects = dataset.objects.len();

    // Calculate the default class (majority class of the entire dataset)
    let all_class_values = dataset.get_class_values(&(0..n_objects).collect::<Vec<_>>());
    let default_class = NominalDataset::get_majority_class(&all_class_values)
        .map(|(class, _, _)| class)
        .unwrap_or_default();

    // Get predictions for each object from concepts
    let mut predictions: HashMap<usize, Prediction> = HashMap::new();

    for (_attr, _value, extent, _intent) in &result.concepts {
        if extent.is_empty() {
            continue;
        }

        // Calculate majority class for this concept
        let class_values = dataset.get_class_values(extent);
        if let Some((majority_class, count, _)) = NominalDataset::get_majority_class(&class_values) {
            let confidence = count as f64 / extent.len() as f64;

            // Assign prediction to each object in extent
            for &obj_idx in extent {
                // Use highest confidence prediction for each object
                let should_update = match predictions.get(&obj_idx) {
                    None => true,
                    Some(existing) => confidence > existing.confidence,
                };

                if should_update {
                    predictions.insert(obj_idx, Prediction {
                        object_idx: obj_idx,
                        predicted_class: majority_class.clone(),
                        confidence,
                    });
                }
            }
        }
    }

    // Count covered objects
    let coverage = predictions.len();

    // Assign default class to uncovered objects (with low confidence)
    for obj_idx in 0..n_objects {
        if !predictions.contains_key(&obj_idx) {
            predictions.insert(obj_idx, Prediction {
                object_idx: obj_idx,
                predicted_class: default_class.clone(),
                confidence: 0.0, // Low confidence for default predictions
            });
        }
    }

    // Collect actual and predicted labels for ALL objects
    let actual: Vec<String> = (0..n_objects)
        .map(|i| dataset.data[i].get(&dataset.class_attribute).cloned().unwrap_or_default())
        .collect();

    let predicted: Vec<String> = (0..n_objects)
        .map(|i| predictions[&i].predicted_class.clone())
        .collect();

    let confidences: Vec<f64> = (0..n_objects)
        .map(|i| predictions[&i].confidence)
        .collect();

    // Build confusion matrix
    let cm = ConfusionMatrix::new(&actual, &predicted);

    // Calculate metrics
    let accuracy = calculate_accuracy(&cm);
    let (macro_precision, macro_recall, macro_f1, per_class) = calculate_macro_metrics(&cm, &actual);
    let mcc = calculate_mcc(&cm);
    let roc_auc = calculate_roc_auc(&actual, &predicted, &confidences, &cm.classes);
    let prc_auc = calculate_prc_auc(&actual, &predicted, &confidences, &cm.classes);

    ClassificationMetrics {
        accuracy,
        macro_precision,
        macro_recall,
        macro_f1,
        mcc,
        roc_auc,
        prc_auc,
        per_class,
        coverage,
        total: n_objects,
    }
}

/// Calculate accuracy
fn calculate_accuracy(cm: &ConfusionMatrix) -> f64 {
    if cm.total == 0 {
        return 0.0;
    }
    let correct: usize = cm.classes.iter()
        .map(|c| cm.true_positives(c))
        .sum();
    correct as f64 / cm.total as f64
}

/// Calculate macro-averaged precision, recall, F1 and per-class metrics
fn calculate_macro_metrics(
    cm: &ConfusionMatrix,
    actual: &[String]
) -> (f64, f64, f64, HashMap<String, PerClassMetrics>) {
    let mut per_class = HashMap::new();
    let mut total_precision = 0.0;
    let mut total_recall = 0.0;
    let mut total_f1 = 0.0;
    let n_classes = cm.classes.len() as f64;

    // Count support for each class
    let mut support_counts: HashMap<String, usize> = HashMap::new();
    for a in actual {
        *support_counts.entry(a.clone()).or_insert(0) += 1;
    }

    for class in &cm.classes {
        let tp = cm.true_positives(class) as f64;
        let fp = cm.false_positives(class) as f64;
        let fn_ = cm.false_negatives(class) as f64;

        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };

        let support = *support_counts.get(class).unwrap_or(&0);

        per_class.insert(class.clone(), PerClassMetrics {
            precision,
            recall,
            f1,
            support,
        });

        total_precision += precision;
        total_recall += recall;
        total_f1 += f1;
    }

    let macro_precision = if n_classes > 0.0 { total_precision / n_classes } else { 0.0 };
    let macro_recall = if n_classes > 0.0 { total_recall / n_classes } else { 0.0 };
    let macro_f1 = if n_classes > 0.0 { total_f1 / n_classes } else { 0.0 };

    (macro_precision, macro_recall, macro_f1, per_class)
}

/// Calculate Matthews Correlation Coefficient
/// For multi-class, uses the generalized MCC formula
fn calculate_mcc(cm: &ConfusionMatrix) -> f64 {
    if cm.total == 0 || cm.classes.len() < 2 {
        return 0.0;
    }

    // For binary classification, use standard formula
    if cm.classes.len() == 2 {
        let class = &cm.classes[0];
        let tp = cm.true_positives(class) as f64;
        let tn = cm.true_negatives(class) as f64;
        let fp = cm.false_positives(class) as f64;
        let fn_ = cm.false_negatives(class) as f64;

        let numerator = tp * tn - fp * fn_;
        let denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();

        if denominator == 0.0 {
            return 0.0;
        }
        return numerator / denominator;
    }

    // For multi-class, use the generalized formula
    // MCC = (c * s - sum(p_k * t_k)) / sqrt((s^2 - sum(p_k^2)) * (s^2 - sum(t_k^2)))
    // where c = sum of correct predictions, s = total samples
    // p_k = sum of predictions for class k, t_k = sum of actual for class k

    let c: f64 = cm.classes.iter()
        .map(|class| cm.true_positives(class) as f64)
        .sum();
    let s = cm.total as f64;

    // p_k = number of times class k was predicted
    let p: Vec<f64> = cm.classes.iter()
        .map(|class| {
            cm.classes.iter()
                .map(|actual| cm.get(actual, class) as f64)
                .sum()
        })
        .collect();

    // t_k = number of times class k was the actual class
    let t: Vec<f64> = cm.classes.iter()
        .map(|class| {
            cm.classes.iter()
                .map(|pred| cm.get(class, pred) as f64)
                .sum()
        })
        .collect();

    let sum_pk_tk: f64 = p.iter().zip(t.iter()).map(|(pk, tk)| pk * tk).sum();
    let sum_pk_sq: f64 = p.iter().map(|pk| pk * pk).sum();
    let sum_tk_sq: f64 = t.iter().map(|tk| tk * tk).sum();

    let numerator = c * s - sum_pk_tk;
    let denom_left = s * s - sum_pk_sq;
    let denom_right = s * s - sum_tk_sq;

    if denom_left <= 0.0 || denom_right <= 0.0 {
        return 0.0;
    }

    numerator / (denom_left * denom_right).sqrt()
}

/// Calculate ROC AUC (macro-averaged for multi-class)
/// Uses one-vs-rest approach
fn calculate_roc_auc(
    actual: &[String],
    predicted: &[String],
    confidences: &[f64],
    classes: &[String],
) -> f64 {
    if classes.len() < 2 || actual.is_empty() {
        return 0.5;
    }

    let mut total_auc = 0.0;
    let mut valid_classes = 0;

    for class in classes {
        // Create binary labels and scores for one-vs-rest
        let binary_actual: Vec<bool> = actual.iter().map(|a| a == class).collect();
        let binary_scores: Vec<f64> = predicted.iter()
            .zip(confidences.iter())
            .map(|(p, &conf)| if p == class { conf } else { 1.0 - conf })
            .collect();

        // Count positives and negatives
        let n_pos = binary_actual.iter().filter(|&&b| b).count();
        let n_neg = binary_actual.iter().filter(|&&b| !b).count();

        if n_pos == 0 || n_neg == 0 {
            continue; // Skip classes with no positive or no negative samples
        }

        // Calculate AUC using trapezoidal rule
        let auc = calculate_binary_roc_auc(&binary_actual, &binary_scores);
        total_auc += auc;
        valid_classes += 1;
    }

    if valid_classes == 0 {
        return 0.5;
    }

    total_auc / valid_classes as f64
}

/// Calculate binary ROC AUC
fn calculate_binary_roc_auc(actual: &[bool], scores: &[f64]) -> f64 {
    // Sort by score descending
    let mut pairs: Vec<(f64, bool)> = scores.iter()
        .zip(actual.iter())
        .map(|(&s, &a)| (s, a))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = actual.iter().filter(|&&b| b).count() as f64;
    let n_neg = actual.iter().filter(|&&b| !b).count() as f64;

    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }

    let mut auc = 0.0;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_tpr = 0.0;
    let mut prev_fpr = 0.0;

    for (_, is_positive) in &pairs {
        if *is_positive {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        let tpr = tp / n_pos;
        let fpr = fp / n_neg;

        // Trapezoidal rule
        auc += (fpr - prev_fpr) * (tpr + prev_tpr) / 2.0;

        prev_tpr = tpr;
        prev_fpr = fpr;
    }

    auc
}

/// Calculate PRC AUC (Precision-Recall Curve Area, macro-averaged)
fn calculate_prc_auc(
    actual: &[String],
    predicted: &[String],
    confidences: &[f64],
    classes: &[String],
) -> f64 {
    if classes.len() < 2 || actual.is_empty() {
        return 0.0;
    }

    let mut total_auc = 0.0;
    let mut valid_classes = 0;

    for class in classes {
        // Create binary labels and scores for one-vs-rest
        let binary_actual: Vec<bool> = actual.iter().map(|a| a == class).collect();
        let binary_scores: Vec<f64> = predicted.iter()
            .zip(confidences.iter())
            .map(|(p, &conf)| if p == class { conf } else { 1.0 - conf })
            .collect();

        let n_pos = binary_actual.iter().filter(|&&b| b).count();
        if n_pos == 0 {
            continue;
        }

        let auc = calculate_binary_prc_auc(&binary_actual, &binary_scores);
        total_auc += auc;
        valid_classes += 1;
    }

    if valid_classes == 0 {
        return 0.0;
    }

    total_auc / valid_classes as f64
}

/// Calculate binary PRC AUC
fn calculate_binary_prc_auc(actual: &[bool], scores: &[f64]) -> f64 {
    // Sort by score descending
    let mut pairs: Vec<(f64, bool)> = scores.iter()
        .zip(actual.iter())
        .map(|(&s, &a)| (s, a))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let n_pos = actual.iter().filter(|&&b| b).count() as f64;

    if n_pos == 0.0 {
        return 0.0;
    }

    let mut auc = 0.0;
    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_recall = 0.0;
    let mut prev_precision = 1.0;

    for (_, is_positive) in &pairs {
        if *is_positive {
            tp += 1.0;
        } else {
            fp += 1.0;
        }

        let precision = tp / (tp + fp);
        let recall = tp / n_pos;

        // Trapezoidal rule
        auc += (recall - prev_recall) * (precision + prev_precision) / 2.0;

        prev_precision = precision;
        prev_recall = recall;
    }

    auc
}

/// Display metrics in a formatted table suitable for academic reporting
pub fn display_metrics_table(metrics: &ClassificationMetrics) {
    println!("\n╔════════════════════════════════════════════════════╗");
    println!("║            Classification Metrics                  ║");
    println!("╠════════════════════════════════════════════════════╣");
    println!("║  Coverage:  {:>6} / {:>6} ({:>6.2}%)              ║",
        metrics.coverage, metrics.total,
        (metrics.coverage as f64 / metrics.total as f64) * 100.0);
    println!("╠════════════════════════════════════════════════════╣");
    println!("║  Accuracy:    {:>11.4}                          ║", metrics.accuracy);
    println!("║  Recall:      {:>11.4}  (macro-averaged)        ║", metrics.macro_recall);
    println!("║  F1-score:    {:>11.4}  (macro-averaged)        ║", metrics.macro_f1);
    println!("║  MCC:         {:>11.4}                          ║", metrics.mcc);
    println!("║  PRC Area:    {:>11.4}                          ║", metrics.prc_auc);
    println!("║  ROC Area:    {:>11.4}                          ║", metrics.roc_auc);
    println!("╚════════════════════════════════════════════════════╝");
}

/// Comparison result between two methods
#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub dataset_name: String,
    pub cnc_metrics: ClassificationMetrics,
    pub cnc_bp_metrics: ClassificationMetrics,
    pub cnc_bp_n: usize,
}

impl ComparisonResult {
    /// Returns which method is better based on F1-score (primary) and MCC (secondary)
    pub fn winner(&self) -> &'static str {
        let cnc_score = self.cnc_metrics.macro_f1 + self.cnc_metrics.mcc * 0.5;
        let bp_score = self.cnc_bp_metrics.macro_f1 + self.cnc_bp_metrics.mcc * 0.5;

        if (cnc_score - bp_score).abs() < 0.001 {
            "Tie"
        } else if cnc_score > bp_score {
            "CNC"
        } else {
            "CNC-BP"
        }
    }
}

/// Display a comparison table between CNC and CNC-BP results
pub fn display_comparison_table(comparisons: &[ComparisonResult]) {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                              CNC vs CNC-BP Comparison Summary                                        ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣");
    println!("║                        │           CNC            │         CNC-BP           │                       ║");
    println!("║ Dataset                │  Acc    F1    MCC   Cov% │  Acc    F1    MCC   Cov% │ Winner                ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣");

    for comp in comparisons {
        let cnc = &comp.cnc_metrics;
        let bp = &comp.cnc_bp_metrics;
        let cnc_cov = (cnc.coverage as f64 / cnc.total as f64) * 100.0;
        let bp_cov = (bp.coverage as f64 / bp.total as f64) * 100.0;

        println!("║ {:22} │ {:5.2} {:5.2} {:5.2} {:5.1}% │ {:5.2} {:5.2} {:5.2} {:5.1}% │ {:21} ║",
            comp.dataset_name,
            cnc.accuracy, cnc.macro_f1, cnc.mcc, cnc_cov,
            bp.accuracy, bp.macro_f1, bp.mcc, bp_cov,
            format!("{} (n={})", comp.winner(), comp.cnc_bp_n)
        );
    }

    println!("╠══════════════════════════════════════════════════════════════════════════════════════════════════════╣");

    // Summary statistics
    let cnc_wins = comparisons.iter().filter(|c| c.winner() == "CNC").count();
    let bp_wins = comparisons.iter().filter(|c| c.winner() == "CNC-BP").count();
    let ties = comparisons.iter().filter(|c| c.winner() == "Tie").count();

    // Average metrics
    let avg_cnc_f1: f64 = comparisons.iter().map(|c| c.cnc_metrics.macro_f1).sum::<f64>() / comparisons.len() as f64;
    let avg_bp_f1: f64 = comparisons.iter().map(|c| c.cnc_bp_metrics.macro_f1).sum::<f64>() / comparisons.len() as f64;
    let avg_cnc_mcc: f64 = comparisons.iter().map(|c| c.cnc_metrics.mcc).sum::<f64>() / comparisons.len() as f64;
    let avg_bp_mcc: f64 = comparisons.iter().map(|c| c.cnc_bp_metrics.mcc).sum::<f64>() / comparisons.len() as f64;

    println!("║ AVERAGE                │      {:6.2} {:5.2}        │      {:6.2} {:5.2}        │  CNC:{} BP:{} Tie:{}     ║",
        avg_cnc_f1, avg_cnc_mcc,
        avg_bp_f1, avg_bp_mcc,
        cnc_wins, bp_wins, ties
    );
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝");

    // Conclusion
    println!("\nConclusion:");
    if bp_wins > cnc_wins {
        println!("  CNC-BP performs better overall ({} wins vs {} for CNC)", bp_wins, cnc_wins);
        println!("  Average F1: CNC={:.4} vs CNC-BP={:.4} (diff: {:+.4})", avg_cnc_f1, avg_bp_f1, avg_bp_f1 - avg_cnc_f1);
        println!("  Average MCC: CNC={:.4} vs CNC-BP={:.4} (diff: {:+.4})", avg_cnc_mcc, avg_bp_mcc, avg_bp_mcc - avg_cnc_mcc);
    } else if cnc_wins > bp_wins {
        println!("  CNC performs better overall ({} wins vs {} for CNC-BP)", cnc_wins, bp_wins);
        println!("  Average F1: CNC={:.4} vs CNC-BP={:.4} (diff: {:+.4})", avg_cnc_f1, avg_bp_f1, avg_cnc_f1 - avg_bp_f1);
        println!("  Average MCC: CNC={:.4} vs CNC-BP={:.4} (diff: {:+.4})", avg_cnc_mcc, avg_bp_mcc, avg_cnc_mcc - avg_bp_mcc);
    } else {
        println!("  Both methods perform similarly ({} wins each)", cnc_wins);
        println!("  Average F1: CNC={:.4} vs CNC-BP={:.4}", avg_cnc_f1, avg_bp_f1);
        println!("  Average MCC: CNC={:.4} vs CNC-BP={:.4}", avg_cnc_mcc, avg_bp_mcc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confusion_matrix() {
        let actual = vec!["A".to_string(), "A".to_string(), "B".to_string(), "B".to_string()];
        let predicted = vec!["A".to_string(), "B".to_string(), "A".to_string(), "B".to_string()];

        let cm = ConfusionMatrix::new(&actual, &predicted);

        assert_eq!(cm.true_positives("A"), 1);
        assert_eq!(cm.false_positives("A"), 1);
        assert_eq!(cm.false_negatives("A"), 1);
        assert_eq!(cm.true_negatives("A"), 1);
    }

    #[test]
    fn test_accuracy() {
        let actual = vec!["A".to_string(), "A".to_string(), "B".to_string(), "B".to_string()];
        let predicted = vec!["A".to_string(), "A".to_string(), "B".to_string(), "A".to_string()];

        let cm = ConfusionMatrix::new(&actual, &predicted);
        let accuracy = calculate_accuracy(&cm);

        assert!((accuracy - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_mcc_perfect() {
        let actual = vec!["A".to_string(), "A".to_string(), "B".to_string(), "B".to_string()];
        let predicted = actual.clone();

        let cm = ConfusionMatrix::new(&actual, &predicted);
        let mcc = calculate_mcc(&cm);

        assert!((mcc - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_mcc_random() {
        // Completely wrong predictions
        let actual = vec!["A".to_string(), "A".to_string(), "B".to_string(), "B".to_string()];
        let predicted = vec!["B".to_string(), "B".to_string(), "A".to_string(), "A".to_string()];

        let cm = ConfusionMatrix::new(&actual, &predicted);
        let mcc = calculate_mcc(&cm);

        assert!((mcc - (-1.0)).abs() < 0.001);
    }
}
