//! Classification rules extraction and manipulation.
//!
//! This module provides structures and functions for working with classification rules
//! extracted from CNC concepts.

use std::collections::{HashMap, HashSet};
use crate::core::{NominalDataset, CncResult};

/// Classification rule extracted from a CNC/CNC-BPC concept.
///
/// A classification rule represents an if-then pattern for classification:
/// **IF** conditions **THEN** class = X (with confidence Y%)
///
/// # Structure
///
/// Each rule contains:
/// - **Conditions**: Attribute-value pairs from the concept's intent
/// - **Predicted class**: The majority class in the concept's extent
/// - **Confidence**: Percentage of the majority class (0-100%)
/// - **Support**: Number of objects covered by the rule
/// - **Coverage**: Percentage of the dataset covered (support/total * 100)
///
/// # Usage
///
/// Rules are typically created by [`extract_rules`] and can be:
/// - Filtered by confidence or support
/// - Sorted by confidence or support
/// - Used to classify new objects via [`matches`](ClassificationRule::matches)
/// - Analyzed with [`get_rules_statistics`]
///
/// # Example
///
/// ```
/// use cnc::{from_arff_auto, cnc};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
/// let rules = ::cnc::extract_rules(&dataset, &result);
///
/// for rule in &rules {
///     println!("{}", rule);
///     // IF outlook=sunny THEN play=no (confidence=100.0%, support=3/14, coverage=21.4%)
/// }
/// # Ok(())
/// # }
/// ```
///
/// # Rule Quality Metrics
///
/// - **Confidence**: Higher is better (indicates rule reliability)
/// - **Support**: Higher = more general, lower = more specific
/// - **Coverage**: Percentage of dataset covered by the rule
/// - **Number of conditions**: Fewer = simpler/general, more = complex/specific
#[derive(Debug, Clone)]
pub struct ClassificationRule {
    /// Conditions (attribute-value pairs from the intent)
    pub conditions: HashMap<String, String>,
    /// Predicted class (majority class in the extent)
    pub predicted_class: String,
    /// Confidence: percentage of the majority class in the extent
    pub confidence: f64,
    /// Support: number of objects covered by this rule
    pub support: usize,
    /// Total objects in dataset
    pub total_objects: usize,
    /// Object indices covered by this rule
    pub covered_objects: Vec<usize>,
    /// Name of the class attribute
    pub class_attribute_name: String,
}

impl ClassificationRule {
    /// Returns the coverage percentage of this rule in the dataset.
    ///
    /// Coverage represents the proportion of objects in the dataset that are
    /// covered by this rule.
    ///
    /// # Example
    ///
    /// ```
    /// # use cnc::{NominalDataset, cnc};
    /// # use std::collections::HashMap;
    /// # let objects = vec!["obj1".to_string()];
    /// # let attributes = vec!["attr".to_string(), "class".to_string()];
    /// # let mut data = vec![];
    /// # let mut obj = HashMap::new();
    /// # obj.insert("attr".to_string(), "val".to_string());
    /// # obj.insert("class".to_string(), "A".to_string());
    /// # data.push(obj);
    /// # let dataset = NominalDataset::new(objects, attributes, "class".to_string(), data);
    /// # let result = cnc(&dataset);
    /// # let rules = ::cnc::extract_rules(&dataset, &result);
    /// for rule in &rules {
    ///     if rule.coverage() > 50.0 {
    ///         println!("High coverage rule: {:.1}%", rule.coverage());
    ///     }
    /// }
    /// ```
    pub fn coverage(&self) -> f64 {
        (self.support as f64 / self.total_objects as f64) * 100.0
    }

    /// Checks if this rule matches a given object.
    ///
    /// A rule matches an object if all conditions (attribute-value pairs) of the rule
    /// are satisfied by the object's attributes.
    ///
    /// # Arguments
    ///
    /// * `object_data` - HashMap containing the object's attribute values
    ///
    /// # Returns
    ///
    /// `true` if all rule conditions match the object, `false` otherwise
    ///
    /// # Example
    ///
    /// ```
    /// # use cnc::{NominalDataset, cnc};
    /// # use std::collections::HashMap;
    /// # let objects = vec!["obj1".to_string()];
    /// # let attributes = vec!["color".to_string(), "class".to_string()];
    /// # let mut data = vec![];
    /// # let mut obj_data = HashMap::new();
    /// # obj_data.insert("color".to_string(), "red".to_string());
    /// # obj_data.insert("class".to_string(), "A".to_string());
    /// # data.push(obj_data.clone());
    /// # let dataset = NominalDataset::new(objects, attributes, "class".to_string(), data);
    /// # let result = cnc(&dataset);
    /// # let rules = ::cnc::extract_rules(&dataset, &result);
    /// // Create a new object to classify
    /// let mut new_object = HashMap::new();
    /// new_object.insert("color".to_string(), "red".to_string());
    /// new_object.insert("size".to_string(), "small".to_string());
    ///
    /// // Find matching rules
    /// for rule in &rules {
    ///     if rule.matches(&new_object) {
    ///         println!("Match! Predicted class: {}", rule.predicted_class);
    ///     }
    /// }
    /// ```
    pub fn matches(&self, object_data: &HashMap<String, String>) -> bool {
        self.conditions.iter().all(|(attr, value)| {
            object_data.get(attr).map(|v| v == value).unwrap_or(false)
        })
    }
}

impl std::fmt::Display for ClassificationRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Format conditions
        if self.conditions.is_empty() {
            write!(f, "IF <no conditions>")?;
        } else {
            write!(f, "IF ")?;
            let mut conds: Vec<_> = self.conditions.iter().collect();
            conds.sort_by_key(|(k, _)| *k);

            for (i, (attr, value)) in conds.iter().enumerate() {
                if i > 0 {
                    write!(f, " AND ")?;
                }
                write!(f, "{}={}", attr, value)?;
            }
        }

        write!(f, " THEN {}={} ", self.class_attribute_name, self.predicted_class)?;
        write!(f, "(confidence={:.1}%, support={}/{}, coverage={:.1}%)",
               self.confidence, self.support, self.total_objects, self.coverage())
    }
}

/// Extracts classification rules from CNC concepts.
///
/// Each concept is transformed into a classification rule where:
/// - The **intent** (common attribute-value pairs) becomes the rule **conditions**
/// - The **majority class** in the extent becomes the **predicted class**
/// - **Confidence** is the percentage of the majority class in the extent
/// - **Support** is the number of objects in the extent
///
/// # Arguments
///
/// * `dataset` - The nominal dataset (used for class attribute name and calculating class distribution)
/// * `result` - CNC result containing the concepts to extract rules from
///
/// # Returns
///
/// A vector of `ClassificationRule`, one for each non-empty concept
///
/// # Rule Structure
///
/// Each extracted rule has the form:
/// ```text
/// IF <conditions> THEN <class> = <value> (confidence=X%, support=N)
/// ```
///
/// # Example
///
/// ```
/// use cnc::{from_arff_auto, cnc, extract_rules};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
/// let rules = extract_rules(&dataset, &result);
///
/// println!("Extracted {} rules from {} concepts", rules.len(), result.concepts.len());
///
/// for rule in &rules {
///     println!("Rule with {} conditions, confidence: {:.1}%",
///              rule.conditions.len(), rule.confidence);
/// }
/// # Ok(())
/// # }
/// ```
pub fn extract_rules(dataset: &NominalDataset, result: &CncResult) -> Vec<ClassificationRule> {
    let mut rules = Vec::new();
    let total_objects = dataset.objects.len();

    for (_pertinent_attr, _attr_value, extent, intent) in &result.concepts {
        if extent.is_empty() {
            continue;
        }

        // Get class distribution in extent
        let class_values = dataset.get_class_values(extent);

        if let Some((majority_class, count, _)) = NominalDataset::get_majority_class(&class_values) {
            let confidence = (count as f64 / extent.len() as f64) * 100.0;

            rules.push(ClassificationRule {
                conditions: intent.clone(),
                predicted_class: majority_class,
                confidence,
                support: extent.len(),
                total_objects,
                covered_objects: extent.clone(),
                class_attribute_name: dataset.class_attribute.clone(),
            });
        }
    }

    rules
}

/// Displays classification rules in a compact, readable format.
///
/// Each rule is printed on a single line showing conditions, predicted class,
/// confidence, support, and coverage percentage.
///
/// # Arguments
///
/// * `rules` - Slice of classification rules to display
///
/// # Output Format
///
/// ```text
/// 3 classification rule(s) extracted:
/// ================================================================================
///
/// Rule 1:
///   IF outlook=sunny THEN play=no (confidence=100.0%, support=3/14, coverage=21.4%)
///
/// Rule 2:
///   IF outlook=overcast THEN play=yes (confidence=100.0%, support=4/14, coverage=28.6%)
/// ================================================================================
/// ```
///
/// # Example
///
/// ```
/// use cnc::{from_arff_auto, cnc, extract_rules, display_rules};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
/// let rules = extract_rules(&dataset, &result);
///
/// display_rules(&rules);
/// # Ok(())
/// # }
/// ```
pub fn display_rules(rules: &[ClassificationRule]) {
    if rules.is_empty() {
        println!("No classification rules extracted");
        return;
    }

    println!("\n{} classification rule(s) extracted:", rules.len());
    println!("{}", "=".repeat(80));

    for (i, rule) in rules.iter().enumerate() {
        println!("\nRule {}:", i + 1);
        println!("  {}", rule);
    }

    println!("\n{}", "=".repeat(80));
}

/// Displays classification rules with comprehensive details and statistics.
///
/// For each rule, this function shows:
/// - All conditions (attribute-value pairs) listed individually
/// - Predicted class with confidence percentage
/// - Coverage statistics (support and percentage of dataset)
/// - Names of all objects covered by the rule
/// - Complete class distribution among covered objects
///
/// # Arguments
///
/// * `dataset` - The dataset (needed for object names and class information)
/// * `rules` - Slice of classification rules to display
///
/// # Output Format
///
/// ```text
/// Rule 1:
///   Conditions (2 attribute(s)):
///     - outlook = sunny
///     - humidity = high
///   Prediction:
///     → class = no (confidence: 100.0%)
///   Coverage:
///     - Support: 3/14 objects (21.4%)
///     - Covered objects: ["obj1", "obj2", "obj8"]
///   Class distribution in covered objects:
///     - no: 3 (100.0%) ← predicted
/// ```
///
/// # Example
///
/// ```
/// use cnc::{from_arff_auto, cnc, extract_rules, display_rules_detailed};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let dataset = from_arff_auto("data-examples/weather.nominal.arff")?;
/// let result = cnc(&dataset);
/// let rules = extract_rules(&dataset, &result);
///
/// display_rules_detailed(&dataset, &rules);
/// # Ok(())
/// # }
/// ```
pub fn display_rules_detailed(dataset: &NominalDataset, rules: &[ClassificationRule]) {
    if rules.is_empty() {
        println!("No classification rules extracted");
        return;
    }

    println!("\n{} classification rule(s) extracted:", rules.len());
    println!("{}", "=".repeat(100));

    for (i, rule) in rules.iter().enumerate() {
        println!("\nRule {}:", i + 1);
        println!("  Conditions ({} attribute(s)):", rule.conditions.len());

        let mut conds: Vec<_> = rule.conditions.iter().collect();
        conds.sort_by_key(|(k, _)| *k);
        for (attr, value) in conds {
            println!("    - {} = {}", attr, value);
        }

        println!("  Prediction:");
        println!("    → class = {} (confidence: {:.1}%)", rule.predicted_class, rule.confidence);

        println!("  Coverage:");
        println!("    - Support: {}/{} objects ({:.1}%)",
                 rule.support, rule.total_objects, rule.coverage());

        // Show covered objects
        let covered_names: Vec<String> = rule.covered_objects.iter()
            .map(|&idx| dataset.objects[idx].clone())
            .collect();
        println!("    - Covered objects: {:?}", covered_names);

        // Show class distribution in covered objects
        let class_values = dataset.get_class_values(&rule.covered_objects);
        let mut class_counts: HashMap<String, usize> = HashMap::new();
        for class_val in &class_values {
            *class_counts.entry(class_val.clone()).or_insert(0) += 1;
        }

        println!("  Class distribution in covered objects:");
        for (class, count) in &class_counts {
            let percentage = (*count as f64 / rule.support as f64) * 100.0;
            let marker = if class == &rule.predicted_class { " ← predicted" } else { "" };
            println!("    - {}: {} ({:.1}%){}", class, count, percentage, marker);
        }
    }

    println!("\n{}", "=".repeat(100));
}

/// Filters rules by minimum confidence threshold.
pub fn filter_rules_by_confidence(rules: &[ClassificationRule], min_confidence: f64) -> Vec<ClassificationRule> {
    rules.iter()
        .filter(|rule| rule.confidence >= min_confidence)
        .cloned()
        .collect()
}

/// Filters rules by minimum support threshold.
pub fn filter_rules_by_support(rules: &[ClassificationRule], min_support: usize) -> Vec<ClassificationRule> {
    rules.iter()
        .filter(|rule| rule.support >= min_support)
        .cloned()
        .collect()
}

/// Sorts rules by confidence in descending order (highest confidence first).
pub fn sort_rules_by_confidence(rules: &mut [ClassificationRule]) {
    rules.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
}

/// Sorts rules by support in descending order (highest support first).
pub fn sort_rules_by_support(rules: &mut [ClassificationRule]) {
    rules.sort_by(|a, b| b.support.cmp(&a.support));
}

/// Computes summary statistics for a set of classification rules.
pub fn get_rules_statistics(rules: &[ClassificationRule]) -> RulesStatistics {
    if rules.is_empty() {
        return RulesStatistics::default();
    }

    let avg_confidence = rules.iter().map(|r| r.confidence).sum::<f64>() / rules.len() as f64;
    let avg_support = rules.iter().map(|r| r.support).sum::<usize>() as f64 / rules.len() as f64;
    let avg_conditions = rules.iter().map(|r| r.conditions.len()).sum::<usize>() as f64 / rules.len() as f64;

    let max_confidence = rules.iter().map(|r| r.confidence).fold(0.0, f64::max);
    let min_confidence = rules.iter().map(|r| r.confidence).fold(100.0, f64::min);

    let max_support = rules.iter().map(|r| r.support).max().unwrap_or(0);
    let min_support = rules.iter().map(|r| r.support).min().unwrap_or(0);

    // Count unique predicted classes
    let unique_classes: HashSet<String> = rules.iter()
        .map(|r| r.predicted_class.clone())
        .collect();

    RulesStatistics {
        total_rules: rules.len(),
        avg_confidence,
        min_confidence,
        max_confidence,
        avg_support,
        min_support,
        max_support,
        avg_conditions,
        unique_predicted_classes: unique_classes.len(),
    }
}

/// Summary statistics computed over a set of classification rules.
#[derive(Debug, Clone, Default)]
pub struct RulesStatistics {
    /// Total number of rules
    pub total_rules: usize,
    /// Average confidence across all rules (percentage)
    pub avg_confidence: f64,
    /// Minimum confidence value (percentage)
    pub min_confidence: f64,
    /// Maximum confidence value (percentage)
    pub max_confidence: f64,
    /// Average support (number of objects covered)
    pub avg_support: f64,
    /// Minimum support value
    pub min_support: usize,
    /// Maximum support value
    pub max_support: usize,
    /// Average number of conditions per rule
    pub avg_conditions: f64,
    /// Number of unique classes predicted by the rules
    pub unique_predicted_classes: usize,
}

impl std::fmt::Display for RulesStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Rules Statistics:")?;
        writeln!(f, "  Total rules: {}", self.total_rules)?;
        writeln!(f, "  Confidence: avg={:.1}%, min={:.1}%, max={:.1}%",
                 self.avg_confidence, self.min_confidence, self.max_confidence)?;
        writeln!(f, "  Support: avg={:.1}, min={}, max={}",
                 self.avg_support, self.min_support, self.max_support)?;
        writeln!(f, "  Avg conditions per rule: {:.1}", self.avg_conditions)?;
        writeln!(f, "  Unique predicted classes: {}", self.unique_predicted_classes)?;
        Ok(())
    }
}
